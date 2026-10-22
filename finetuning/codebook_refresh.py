"""Bounded train-only dead-code refresh; no change to code count or stream rate."""
import torch


class CodebookRefresh:
    def __init__(self, model, optimizer, seed, age=3, cap=32):
        self.model, self.optimizer, self.seed = model, optimizer, seed
        self.age, self.cap = age, cap
        self.inactive_epochs = torch.zeros(model.num_embeddings, dtype=torch.long)

    def begin_epoch(self, epoch):
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
        self.used = torch.zeros(self.model.num_embeddings, dtype=torch.bool)
        self.reservoir = []
        self.step = 0

    def observe(self, latent, indices):
        # All usages count; a bounded sample of TRAIN encoder latents is retained.
        self.used[torch.unique(indices.detach()).cpu()] = True
        if self.step % 64 == 0:
            flat = latent.detach().permute(0, 2, 3, 1).reshape(-1, self.model.embedding_dim)
            chosen = torch.randperm(len(flat), generator=self.generator)[:128].to(flat.device)
            values = flat[chosen].cpu()
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite training latents in refresh reservoir")
            if len(self.reservoir) < 64:
                self.reservoir.append(values)
        self.step += 1

    def refresh(self):
        self.inactive_epochs = torch.where(self.used, 0, self.inactive_epochs + 1)
        dead = torch.where(self.inactive_epochs >= self.age)[0]
        if not len(dead) or not self.reservoir:
            return 0
        # Rotate selection deterministically rather than favor low index IDs.
        dead = dead[torch.randperm(len(dead), generator=self.generator)[:self.cap]]
        vectors = torch.cat(self.reservoir)
        selected = torch.randperm(len(vectors), generator=self.generator)[:len(dead)]
        dead = dead[:len(selected)]
        parameter = self.model.quantizer.embedding.weight
        ids = dead.to(parameter.device)
        with torch.no_grad():
            parameter[ids] = vectors[selected].to(parameter.device)
            # Do not carry stale Adam momentum into newly seeded embeddings.
            state = self.optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state:
                    state[name][ids] = 0
        self.inactive_epochs[dead] = 0
        return len(dead)

    def state_dict(self):
        return {"inactive_epochs": self.inactive_epochs.clone(), "age": self.age, "cap": self.cap}

    def load_state_dict(self, state):
        if state["age"] != self.age or state["cap"] != self.cap or state["inactive_epochs"].numel() != self.model.num_embeddings:
            raise ValueError("Codebook refresh state/policy mismatch")
        self.inactive_epochs = state["inactive_epochs"].long().cpu().clone()
