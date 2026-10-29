"""Conservative spatial rejection of persistently excessive channel amplitudes.

This is a coverage-selection heuristic, not denoising or physical calibration.
Retained samples are never clipped, rescaled, interpolated, or joined across gaps.
"""
import hashlib
import json

import numpy as np

POLICY = "persistent-amplitude-v1"
DEFAULTS = {
    "amplitude_ratio": 8.0,
    "persistence_fraction": .8,
    "window_samples": 1000,
    "windows_per_block": 8,
    "minimum_windows": 4,
    "minimum_run_channels": 2,
    "maximum_removed_fraction": .25,
    "minimum_reference_channels": 8,
}


def contiguous_regions(mask):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return [part for part in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1) if part.size]


def detect_spatial_bands(blocks, **overrides):
    """Use bounded, evenly spaced nonoverlapping windows; one decision per file.

    Input is [block, original channel, time]. Each window uses temporal MAD plus
    robust offset magnitude and a spatial median reference. Sparse peaks cannot
    dominate a channel's scale.
    Parameters are fixed; no labels, reconstruction errors or cross-file fitting.
    """
    params = {**DEFAULTS, **overrides}
    if set(overrides)-set(DEFAULTS):
        raise ValueError("Unknown spatial band filter parameter")
    if (params["amplitude_ratio"] <= 1 or not 0 < params["persistence_fraction"] <= 1
            or not 0 < params["maximum_removed_fraction"] < .5
            or any(params[key] < 1 or int(params[key]) != params[key] for key in (
                "window_samples", "windows_per_block", "minimum_windows", "minimum_run_channels", "minimum_reference_channels"))):
        raise ValueError("Invalid spatial band filter parameters")
    values = np.asarray(blocks)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("Expected nonempty [block,channel,time] data")
    channels = values.shape[1]
    votes = np.zeros(channels, dtype=np.int64)
    observations = np.zeros(channels, dtype=np.int64)
    windows = []
    eligible_windows = 0
    width = int(params["window_samples"])
    for block in range(values.shape[0]):
        available = values.shape[2]//width
        if not available:
            continue
        slots = np.unique(np.linspace(0, available-1, min(available, int(params["windows_per_block"])), dtype=int))
        for slot in slots:
            start = int(slot*width)
            raw = values[block, :, start:start+width]
            finite = np.isfinite(raw).all(axis=1)
            indices = np.flatnonzero(finite)
            if len(indices) < params["minimum_reference_channels"]:
                continue
            part = raw[indices].astype(np.float64)
            center = np.median(part, axis=1)
            scale = 1.4826*np.median(np.abs(part-center[:, None]), axis=1)
            scale = np.maximum(scale, np.abs(center))
            # Include zero scales in the reference: if most channels are constant,
            # there is insufficient evidence to label the few active ones invalid.
            reference = float(np.median(scale))
            windows.append({"block": block, "start": start, "end": start+width,
                            "reference_scale": reference})
            if not np.isfinite(reference) or reference <= 0:
                continue
            eligible_windows += 1
            observations[indices] += 1
            votes[indices] += scale > params["amplitude_ratio"]*reference
    fraction = np.divide(votes, observations, out=np.zeros(channels), where=observations>0)
    candidate = (observations >= params["minimum_windows"]) & (fraction >= params["persistence_fraction"])
    removed = np.zeros(channels, dtype=bool)
    for region in contiguous_regions(candidate):
        if len(region) >= params["minimum_run_channels"]:
            removed[region] = True
    observed = observations >= params["minimum_windows"]
    status = "selected" if removed.any() else "no_persistent_excess"
    if eligible_windows < params["minimum_windows"]:
        status = "abstained_insufficient_windows"
        removed[:] = False
    elif removed.sum()/max(1, observed.sum()) > params["maximum_removed_fraction"]:
        status = "abstained_excessive_coverage_loss"
        removed[:] = False
    result = {"policy": POLICY, "parameters": params, "status": status,
              "source_channels": channels, "sampled_windows": windows,
              "eligible_windows": eligible_windows,
              "removed_source_channel_indices": np.flatnonzero(removed).tolist(),
              "candidate_source_channel_indices": np.flatnonzero(candidate).tolist(),
              "observations_by_channel": observations.tolist(), "votes_by_channel": votes.tolist(),
              "rule_note": "Relative persistent amplitude heuristic; no physical validity guarantee. Abstention keeps all amplitude candidates."}
    result["filter_id"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def retained_regions(block, decision, max_channels=None):
    """Return contiguous source indices, optionally tiled without losing valid rows."""
    block = np.asarray(block)
    if block.ndim != 2 or block.shape[0] != decision["source_channels"]:
        raise ValueError("Source geometry differs from filter decision")
    mask = np.isfinite(block).all(axis=1)
    mask[np.asarray(decision["removed_source_channel_indices"], dtype=int)] = False
    regions = contiguous_regions(mask)
    if max_channels is not None:
        if max_channels < 1:
            raise ValueError("Positive spatial tile limit required")
        regions = [region[i:i+max_channels] for region in regions for i in range(0, len(region), max_channels)]
    return regions
