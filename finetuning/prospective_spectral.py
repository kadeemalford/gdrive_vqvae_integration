"""Descriptive pooled band energies and phase with explicit undefined coverage."""
import numpy as np


def spectral_statistics(target, prediction, dt_s, reference_scale):
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    if (target.shape != prediction.shape or target.ndim != 2 or target.shape[1] < 2
            or not np.isfinite(target).all() or not np.isfinite(prediction).all()
            or not np.isfinite(dt_s) or dt_s <= 0 or reference_scale <= 0):
        raise ValueError("Finite matching physical windows and positive scales required")
    window = np.hanning(target.shape[1])
    x = np.fft.rfft((target - target.mean(axis=1, keepdims=True)) * window, axis=1)
    y = np.fft.rfft((prediction - prediction.mean(axis=1, keepdims=True)) * window, axis=1)
    frequency = np.fft.rfftfreq(target.shape[1], dt_s)
    floor = reference_scale ** 2 * float((window ** 2).sum())
    result = {}
    for name, lower, upper in (("0_1khz", 0, 1000), ("1_5khz", 1000, 5000),
                               ("5_10khz", 5000, 10001)):
        mask = (frequency >= lower) & (frequency < upper)
        a, b = x[:, mask], y[:, mask]
        pa, pb = np.abs(a) ** 2, np.abs(b) ** 2
        defined = (pa > floor) & (pb > floor)
        weight = pa[defined]
        phase_error = np.abs(np.angle(b[defined] * a[defined].conjugate()))
        result[name] = {"target_energy": float(pa.sum()), "error_energy": float((np.abs(b - a) ** 2).sum()),
                        "prediction_energy": float(pb.sum()), "phase_weight": float(weight.sum()),
                        "phase_error_weighted_sum": float((weight * phase_error).sum()),
                        "phase_defined_bins": int(defined.sum()), "total_bins": int(pa.size)}
    return result


def merge_spectral(total, addition):
    for band, values in addition.items():
        target = total.setdefault(band, {key: 0 for key in values})
        for key, value in values.items():
            target[key] += value


def spectral_result(total):
    result = {}
    for band, values in total.items():
        energy, weight = values["target_energy"], values["phase_weight"]
        result[band] = {**values,
                        "relative_rmse": float(np.sqrt(values["error_energy"] / energy)) if energy > 0 else None,
                        "rms_ratio": float(np.sqrt(values["prediction_energy"] / energy)) if energy > 0 else None,
                        "phase_error_radians": values["phase_error_weighted_sum"] / weight if weight > 0 else None}
    return result
