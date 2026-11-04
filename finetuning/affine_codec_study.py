"""Shared physical scoring and frozen decoder for adaptive-restoration studies."""
import numpy as np
import torch

from compression.affine_restoration import restore
from compression.numeric_restoration import parse_input_side
from finetuning.diagnose_reconstruction import spectral_metrics
from finetuning.evaluate_checkpoints import metrics
from finetuning.physical_validation import ordinary_fidelity
from finetuning.reversible_normalization import parse_side_stream
from finetuning.paired_search_diagnostics import finite_json
from finetuning.run_index_search import vectors


def normalized_restoration(restored, packet):
    _, mean, scale, _, _ = parse_input_side(packet.side_stream)
    normalized = np.zeros(restored.shape, dtype=np.float32)
    normalized[packet.valid] = ((restored[packet.valid].astype(np.float64) - mean) / scale).astype(np.float32)
    return normalized


def score_restored(row, raw, packet, restored):
    if restored.shape != raw.shape or not np.isfinite(restored[packet.valid]).all():
        raise ValueError('Finite restored array on the original shape required')
    assert raw[~packet.valid].tobytes() == restored[~packet.valid].tobytes()
    normalized = normalized_restoration(restored, packet)
    return finite_json({'family': row['family'], 'configuration': row['configuration'],
        '_aggregation_weight': float(raw.size),
        **metrics(packet.values[packet.valid], normalized[packet.valid]),
        **spectral_metrics(packet.values, normalized), **ordinary_fidelity(raw, restored, packet.valid)})


def temporal_distortion(restored, packet):
    difference = normalized_restoration(restored, packet)[packet.valid].astype(np.float64) - packet.values[packet.valid]
    return float(.5 * np.mean(difference ** 2) + .5 * np.mean(np.abs(difference)))


def decode_affine_frame(frame, stream, model, device):
    indices, shape, latent, side = stream.decode(frame)
    with torch.no_grad():
        prediction = model.decoder(vectors(torch.from_numpy(indices.astype(np.int64)).to(device),
            model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]][0, 0].cpu().numpy()
    return restore(prediction, side)
