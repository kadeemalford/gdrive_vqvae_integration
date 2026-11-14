"""Versioned joint CNN/context VQ codec; no continuous skips across quantization."""
import torch
import torch.nn.functional as F

from finetuning.latent_context_adapter import LatentContextAdapter
from finetuning.model_factory import build_model, new_model
from finetuning.run_index_search import nearest, vectors
from finetuning.update_schedule import load_independent_optimizer_state

VERSION = 'joint-context-vq-v1'
ARMS = ('pointwise', 'axial')


class JointContextVQ(torch.nn.Module):
    def __init__(self, base, kind):
        super().__init__()
        if kind not in ARMS:
            raise ValueError('Unknown joint-context architecture')
        self.base, self.kind = base, kind
        # Both controls receive the same known dimensions and positional features.
        context_kind = 'shape' if kind == 'pointwise' else 'axial'
        self.analysis_context = LatentContextAdapter(context_kind)
        self.synthesis_context = LatentContextAdapter(context_kind)
        self.downsample_levels = base.downsample_levels
        self.num_embeddings = base.num_embeddings
        self.latent_geometry = base.latent_geometry

    def analyze(self, target):
        h, w = target.shape[-2:]
        features = self.base.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
        return self.analysis_context(features, (h, w))

    def encode_indices(self, target):
        features = self.analyze(target)
        return nearest(features, self.base.quantizer.embedding.weight), tuple(features.shape[-2:])

    def decode_indices(self, indices, latent_shape, original_shape):
        quantized = vectors(indices, self.base.quantizer.embedding.weight, latent_shape)
        return self.synthesize(quantized, original_shape)

    def synthesize(self, quantized, shape):
        contextual = self.synthesis_context(quantized, shape)
        return self.base.decoder(contextual)[..., :shape[0], :shape[1]]

    def forward(self, target):
        features = self.analyze(target)
        quantized, loss, indices, perplexity = self.base.quantizer(features)
        if not self.training:
            quantized = vectors(indices, self.base.quantizer.embedding.weight, features.shape[-2:])
        return self.synthesize(quantized, target.shape[-2:]), loss, indices, perplexity


def context_config(base_config, kind):
    if (kind not in ARMS or base_config.get('architecture', 'native-v1') != 'native-v1'
            or base_config.get('latent_geometry') != 'spatial-4x'
            or base_config.get('embedding_dim') != 128 or base_config.get('num_embeddings') != 2048
            or base_config.get('downsample_levels') != 4):
        raise ValueError('Expected original compact K2048 native backbone and known context arm')
    return {'architecture': VERSION, 'context_kind': kind,
            'base_config': {**base_config, 'exact_eval_quantization': True, 'activation_checkpointing': True}}


def initialize_joint_context(source, kind, device):
    config = context_config(source['config'], kind)
    base = build_model(config['base_config'], source, device)
    return JointContextVQ(base, kind).to(device), config


def build_joint_context(config, saved, device):
    if config.get('architecture') != VERSION or config != saved['config']:
        raise ValueError('Joint-context checkpoint architecture/configuration differs')
    context_config(config['base_config'], config['context_kind'])
    base = new_model(config['base_config'], device)
    model = JointContextVQ(base, config['context_kind']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.eval()


def inherited_joint_optimizer(model, source_state):
    optimizer = torch.optim.AdamW(model.base.parameters(), lr=1e-4, weight_decay=1e-5)
    load_independent_optimizer_state(optimizer, source_state)
    for group in optimizer.param_groups:
        group['lr'] = 1e-4
    added = [*model.analysis_context.parameters(), *model.synthesis_context.parameters()]
    context_group = {key: value for key, value in optimizer.param_groups[0].items() if key != 'params'}
    context_group['params'] = added
    optimizer.add_param_group(context_group)
    return optimizer


def assert_joint_optimizer_steps(model, optimizer, completed):
    base = {float(optimizer.state[p]['step']) for p in model.base.parameters() if p in optimizer.state}
    context = {float(optimizer.state[p]['step']) for p in
               [*model.analysis_context.parameters(), *model.synthesis_context.parameters()] if p in optimizer.state}
    if base != {98304. + completed} or context != ({float(completed)} if completed else set()):
        raise ValueError('Joint backbone/context optimizer counter mismatch')
    if completed and len(optimizer.state) != sum(1 for _ in model.parameters()):
        raise ValueError('Some joint-context parameters did not participate in training')
