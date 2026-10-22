"""Restore either representation, then score on one common global-normalized target."""
import math
import numpy as np

try:
    from .latent_geometry import latent_positions
    from .reversible_normalization import encode, decode, parse_side_stream
    from .channel_normalization import encode_channel, decode_channel
    from .robust_escape import encode_robust
except ImportError:
    from latent_geometry import latent_positions
    from reversible_normalization import encode, decode, parse_side_stream
    from channel_normalization import encode_channel, decode_channel
    from robust_escape import encode_robust


def ordinary_fidelity(raw, restored, valid):
    raw, restored = raw.astype(np.float64), restored.astype(np.float64)
    original, predicted = raw[valid], restored[valid]
    centered = original - original.mean()
    error = predicted - original
    rms = math.sqrt(float(np.mean(centered ** 2)))
    rmse = math.sqrt(float(np.mean(error ** 2)))
    corr = float(np.corrcoef(original, predicted)[0, 1]) if rms > 0 and predicted.std() > 0 else 0.0
    channel_errors, channel_correlations, nonweak_errors = [], [], []
    weak_channels = 0
    diagnostic_floor = max(1e-6, rms * 1e-6)
    for channel in range(raw.shape[0]):
        mask = valid[channel]
        if not mask.any():
            continue
        reference, reconstruction = raw[channel, mask], restored[channel, mask]
        scale = float(reference.std())
        if scale <= 1e-6:
            continue
        relative_error = float(np.sqrt(np.mean((reconstruction - reference) ** 2)) / scale)
        channel_errors.append(relative_error)
        if scale < diagnostic_floor:
            weak_channels += 1
        else:
            nonweak_errors.append(relative_error)
        channel_correlations.append(float(np.corrcoef(reference, reconstruction)[0, 1]) if reconstruction.std() > 0 else 0.)
    return {"ordinary_global_centered_relative_rmse": rmse / (rms + 1e-6), "ordinary_correlation": corr,
            "ordinary_channel_macro_relative_rmse": float(np.mean(channel_errors)) if channel_errors else 0.,
            "ordinary_channel_macro_correlation": float(np.mean(channel_correlations)) if channel_correlations else 0.,
            "ordinary_channel_median_relative_rmse": float(np.median(channel_errors)) if channel_errors else 0.,
            "ordinary_channel_p90_relative_rmse": float(np.percentile(channel_errors, 90)) if channel_errors else 0.,
            "ordinary_nonweak_channel_macro_relative_rmse": float(np.mean(nonweak_errors)) if nonweak_errors else float("nan"),
            "diagnostic_weak_variable_channels": weak_channels, "diagnostic_channel_std_floor": diagnostic_floor,
            "variable_ordinary_channels": len(channel_errors), "ordinary_raw_rmse": rmse}


def restore_for_validation(path, predicted, policy, model, detailed_channels=False):
    with np.load(path, allow_pickle=False) as payload:
        raw = np.asarray(payload["data"], dtype=np.float32)
    common = encode_robust(raw) if policy == "robust-escape-v1" else encode(raw)
    if policy == "escape-v1":
        packet, restore = common, decode
    elif policy == "channel-escape-v1":
        packet, restore = encode_channel(raw), decode_channel
    elif policy == "robust-escape-v1":
        packet, restore = common, decode
    else:
        raise ValueError("Physical validation requires an escape policy")
    restored = restore(predicted, packet.side_stream)
    if raw[~packet.valid].tobytes() != restored[~packet.valid].tobytes():
        raise AssertionError("Escape restoration not bit exact")
    _, mean, scale, _, _ = parse_side_stream(common.side_stream)
    normalized_prediction = ((restored.astype(np.float64) - mean) / scale).astype(np.float32)
    normalized_prediction[~packet.valid] = 0
    physical = ordinary_fidelity(raw, restored, packet.valid)
    positions = latent_positions(model, *raw.shape)
    bits = positions * math.ceil(math.log2(model.num_embeddings))
    extras = {"ordinary_channel_macro_relative_rmse": physical["ordinary_channel_macro_relative_rmse"],
              "ordinary_channel_macro_correlation": physical["ordinary_channel_macro_correlation"],
              "side_bits_per_cell": 8 * len(packet.side_stream) / raw.size,
              "index_plus_side_nominal_ratio": 32 * raw.size / (bits + 8 * len(packet.side_stream))}
    if detailed_channels:
        for key in ("ordinary_channel_median_relative_rmse", "ordinary_channel_p90_relative_rmse",
                    "ordinary_nonweak_channel_macro_relative_rmse", "diagnostic_weak_variable_channels",
                    "diagnostic_channel_std_floor", "variable_ordinary_channels", "ordinary_raw_rmse"):
            extras[key] = physical[key]
    return common.values, normalized_prediction, extras
