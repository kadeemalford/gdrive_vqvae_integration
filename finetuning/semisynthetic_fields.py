"""Declared synthetic gauge/time response added to complete measured train carriers."""
import numpy as np


def compose(raw, dt, dx, reference_scale, rng):
    raw = np.asarray(raw)
    if (raw.dtype != np.float32 or raw.ndim != 2 or not raw.size or not np.isfinite(raw).all()
            or not np.isfinite([dt, dx, reference_scale]).all()
            or min(dt, dx, reference_scale) <= 0):
        raise ValueError("Invalid finite physical carrier or geometry")
    height, width = raw.shape
    x = np.arange(height, dtype=np.float64)[:, None] * dx
    t = np.arange(width, dtype=np.float64)[None] * dt
    gauge = float(rng.choice([5., 6.25, 25.])) * dx
    components = []
    for index in range(3):
        components.append({
            "f0": float(rng.uniform(.005, .35) / dt),
            "f1": float(rng.uniform(.005, .35) / dt),
            "wavelength": float(np.exp(rng.uniform(np.log(2.5), np.log(128)))) * dx,
            "direction": float(rng.choice([-1., 1.])),
            "phase": float(rng.uniform(-np.pi, np.pi)),
            "weight": float(rng.uniform(.25, 1.)),
            "center": float(rng.uniform(0, max(width - 1, 1)) * dt),
            "width": float(rng.uniform(8, max(9, width / 3)) * dt),
            "transient": index == 2})

    def field(position, time):
        result = np.zeros((height, width), np.float64)
        for item in components:
            slope = (item["f1"] - item["f0"]) / max(width * dt, dt)
            phase = (2 * np.pi * (item["f0"] * time + .5 * slope * time**2
                     - item["direction"] * position / item["wavelength"]) + item["phase"])
            envelope = np.exp(-.5 * ((time - item["center"]) / item["width"])**2) if item["transient"] else 1.
            result += item["weight"] * np.sin(phase) * envelope
        return result

    response = ((field(x + gauge / 2, t + dt / 2) - field(x - gauge / 2, t + dt / 2))
                - (field(x + gauge / 2, t - dt / 2) - field(x - gauge / 2, t - dt / 2))) / (gauge * dt)
    response -= response.mean()
    rms = float(np.sqrt(np.square(response).mean()))
    amplitude_ratio = float(rng.uniform(.25, 1.))
    amplitude = amplitude_ratio * max(float(raw.astype(np.float64).std()), reference_scale)
    addition = response * (amplitude / rms) if rms > 0 else np.zeros_like(response)
    with np.errstate(over="ignore", invalid="ignore"):
        result = (raw.astype(np.float64) + addition).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Synthetic composition overflow; no clipping permitted")
    return result, {
        "gauge_m_synthetic": gauge, "amplitude_ratio": amplitude_ratio,
        "added_rms_float64": float(np.sqrt(np.square(addition).mean())),
        "carrier_std_float64": float(raw.astype(np.float64).std()),
        "components": components,
        "limits": "Complete measured carrier; synthetic response is not calibrated FEBUS physics."}
