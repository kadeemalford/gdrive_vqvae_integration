"""Paired exact-code search with/without encoder-adaptive affine restoration."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from compression.affine_restoration import candidates, restore
from finetuning.affine_codec_study import score_restored, temporal_distortion, decode_affine_frame
from finetuning.representation_study import distortion
from finetuning.run_index_search import nearest, vectors
from finetuning.run_physical_guard_search import physical_admit


def paired_search(model, tensor, valid, row, raw, packet, stream, learning_rate, steps=48, bounded_ols=False):
    """Share exactly one latent proposal trajectory; evaluate a nested candidate set.

    All chosen outputs are decoded from hard indices and transmitted side bytes.
    Timing is combined for both selectors, not an isolated latency comparison.
    """
    if steps < 1 or learning_rate <= 0:
        raise ValueError('Positive search budget required')
    h, w = tensor.shape[-2:]
    with torch.no_grad():
        z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
        embedding = model.quantizer.embedding.weight.detach()
        original = nearest(z, embedding)
        latent = tuple(z.shape[-2:])
        initial = vectors(original, embedding, latent)
        array = model.decoder(initial)[..., :h, :w][0, 0].cpu().numpy()
        restored = restore(array, packet.side_stream)
        before = score_restored(row, raw, packet, restored)
        source_frame = stream.encode(original, (h, w), latent, packet.side_stream)
        source_loss = temporal_distortion(restored, packet)
        costs = -torch.log2(torch.from_numpy(stream.probabilities).to(tensor.device))
    selectors = ('strict', 'affine', 'bounded') if bounded_ols else ('strict', 'affine')
    best = {name: {'loss': source_loss, 'frame': source_frame, 'metrics': before,
                   'step': 0, 'restoration': 'original', 'changed_indices': False,
                   'gain': 1., 'offset': 0.} for name in selectors}
    evaluated = {'plain': 0, 'affine': 0, 'over_budget': 0}

    def consider(indices, array, step):
        frame = stream.encode(indices, (h, w), latent, packet.side_stream)
        if len(frame) > len(source_frame):
            evaluated['over_budget'] += 1
            return
        restored = restore(array, packet.side_stream)
        metric = score_restored(row, raw, packet, restored)
        loss = temporal_distortion(restored, packet)
        evaluated['plain'] += 1
        changed = not torch.equal(indices, original)
        if physical_admit(before, metric):
            for name in selectors:
                if loss < best[name]['loss']:
                    best[name] = {'loss': loss, 'frame': frame, 'metrics': metric,
                                  'step': step, 'restoration': 'original', 'changed_indices': changed,
                                  'gain': 1., 'offset': 0.}
        for fit in candidates(packet.values, array, packet.valid, packet.side_stream,
                              baseline_rms=before['reconstruction_rms_ratio'] if bounded_ols else None):
            after_array = restore(array, fit['side'])
            after = score_restored(row, raw, packet, after_array)
            value = temporal_distortion(after_array, packet)
            evaluated['affine'] += 1
            eligible_selectors = (('bounded',) if fit['kind'] == 'bounded_ols' else
                                  ('affine', 'bounded') if bounded_ols else ('affine',))
            if any(value < best[name]['loss'] for name in eligible_selectors) and physical_admit(before, after):
                fitted_frame = stream.encode(indices, (h, w), latent, fit['side'])
                assert len(fitted_frame) == len(frame)
                for name in eligible_selectors:
                    if value < best[name]['loss']:
                        best[name] = {'loss': value, 'frame': fitted_frame, 'metrics': after,
                                     'step': step, 'restoration': fit['kind'], 'changed_indices': changed,
                                     'gain': fit['gain'], 'offset': fit['offset']}
    consider(original, array, 0)
    proposal = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([proposal], lr=learning_rate)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = model.decoder(proposal)[..., :h, :w]
        loss = distortion(predicted, tensor, valid)
        loss.backward()
        if not torch.isfinite(proposal.grad).all():
            raise RuntimeError('Nonfinite affine-search proposal gradient')
        optimizer.step()
        if step % 4 == 0 or step == steps:
            with torch.no_grad():
                indices = nearest(proposal, embedding)
                delta = costs[indices] - costs[original]
                cheap = delta <= 0
                limited = torch.where(cheap, indices, original)
                available = -float(delta[cheap].sum())
                positive = torch.where(delta > 0)[0]
                order = positive[torch.argsort(delta[positive])]
                prefix = torch.cumsum(delta[order], 0) <= max(0., available - 32.)
                limited[order[prefix]] = indices[order[prefix]]
                for candidate in (indices, limited):
                    array = model.decoder(vectors(candidate, embedding, latent))[..., :h, :w][0, 0].cpu().numpy()
                    consider(candidate, array, step)
    for name, item in best.items():
        assert len(item['frame']) <= len(source_frame)
        replay = decode_affine_frame(item['frame'], stream, model, tensor.device)
        assert score_restored(row, raw, packet, replay) == item['metrics']
        if item['frame'] != source_frame:
            assert physical_admit(before, item['metrics'])
    assert best['affine']['loss'] <= best['strict']['loss']
    if bounded_ols:
        assert best['bounded']['loss'] <= best['affine']['loss']
    return source_frame, before, best, evaluated
