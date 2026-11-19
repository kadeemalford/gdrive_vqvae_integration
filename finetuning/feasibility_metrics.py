"""Target-anchored spectral diagnostics for the prospective feasibility contract."""
import numpy as np


VERSION = "feasibility-target-support-v1"
BANDS = (("0_1khz", 0., 1000.), ("1_5khz", 1000., 5000.),
         ("5_10khz", 5000., 10000.))


def target_supported_spectra(target, prediction, dt_s, reference_scale):
    """Return additive statistics, with support selected only from the target.

    This version describes demeaned, Hann-windowed AC at 20 kHz. DC bias belongs
    in the physical time-domain audit. Missing reconstructed phase receives pi
    only in an explicitly named conservative penalty, never as a measured angle.
    The scale is a frozen diagnostic threshold, not an instrument noise floor.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (target.ndim != 2 or target.shape != prediction.shape or not target.size
            or target.shape[1] < 4 or not np.isfinite(target).all()
            or not np.isfinite(prediction).all() or not np.isfinite(dt_s)
            or not np.isclose(dt_s, 1 / 20000, rtol=1e-10, atol=0)
            or not np.isfinite(reference_scale) or reference_scale <= 0):
        raise ValueError("Expected finite matching 20 kHz windows and a positive frozen scale")
    window = np.hanning(target.shape[1])
    x = np.fft.rfft((target - target.mean(axis=1, keepdims=True)) * window, axis=1)
    y = np.fft.rfft((prediction - prediction.mean(axis=1, keepdims=True)) * window, axis=1)
    frequency = np.fft.rfftfreq(target.shape[1], dt_s)
    weights = np.full(frequency.size, 2., dtype=np.float64)
    weights[0] = 1.
    if target.shape[1] % 2 == 0:
        weights[-1] = 1.
    floor = reference_scale ** 2 * float(window @ window)
    result = {}
    for name, lower, upper in BANDS:
        mask = (frequency > 0) & (frequency >= lower)
        mask &= frequency <= upper if upper == 10000 else frequency < upper
        a, b = x[:, mask], y[:, mask]
        weight = weights[mask]
        pa, pb = np.abs(a) ** 2, np.abs(b) ** 2
        support = pa > floor
        observed = support & (pb > floor)
        missing = support & ~observed
        target_energy = pa * weight
        error_energy = np.abs(b - a) ** 2 * weight
        angles = np.abs(np.angle(b[observed] * a[observed].conjugate()))
        phase_sum = float((target_energy[observed] * angles).sum())
        missing_energy = float(target_energy[missing].sum())
        result[name] = {
            "total_bins": int(pa.size),
            "supported_bins": int(support.sum()),
            "observed_phase_bins": int(observed.sum()),
            "missing_phase_bins": int(missing.sum()),
            "target_energy": float(target_energy.sum()),
            "prediction_energy": float((pb * weight).sum()),
            "error_energy": float(error_energy.sum()),
            "supported_target_energy": float(target_energy[support].sum()),
            "supported_error_energy": float(error_energy[support].sum()),
            "supported_amplitude_error_energy": float(
                (((np.abs(b) - np.abs(a)) ** 2 * weight)[support]).sum()),
            "observed_phase_target_energy": float(target_energy[observed].sum()),
            "missing_phase_target_energy": missing_energy,
            "observed_phase_weighted_sum": phase_sum,
            "phase_penalty_weighted_sum_missing_pi": phase_sum + np.pi * missing_energy,
        }
    return result


def summarize_target_spectra(statistics):
    """Finalize after summing additive statistics for the same frozen contract."""
    result = {}
    for band, values in statistics.items():
        energy = values["target_energy"]
        supported = values["supported_target_energy"]
        observed = values["observed_phase_target_energy"]
        result[band] = {
            **values,
            "relative_complex_rmse": float(np.sqrt(values["error_energy"] / energy))
            if energy > 0 else None,
            "rms_ratio": float(np.sqrt(values["prediction_energy"] / energy))
            if energy > 0 else None,
            "target_support_energy_fraction": supported / energy if energy > 0 else None,
            "supported_complex_rmse": float(np.sqrt(values["supported_error_energy"] / supported))
            if supported > 0 else None,
            "supported_amplitude_rmse": float(np.sqrt(
                values["supported_amplitude_error_energy"] / supported))
            if supported > 0 else None,
            "missing_phase_energy_fraction": values["missing_phase_target_energy"] / supported
            if supported > 0 else None,
            "observed_phase_error_radians": values["observed_phase_weighted_sum"] / observed
            if observed > 0 else None,
            "phase_penalty_radians_missing_pi": values["phase_penalty_weighted_sum_missing_pi"] / supported
            if supported > 0 else None,
        }
    return result
