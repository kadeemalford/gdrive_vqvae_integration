"""Explicit source-index maps for the existing finite-channel/central-crop policy."""
import hashlib
import json

import numpy as np


def selected_indices(finite, max_channels):
    finite = np.asarray(finite, dtype=bool)
    if finite.ndim != 1 or max_channels < 1:
        raise ValueError("Expected channel validity vector and positive channel limit")
    indices = np.flatnonzero(finite)
    count = min(max_channels, len(indices))
    start = max(0, (len(indices) - count) // 2)
    return indices[start:start + count]


def spatial_metadata(indices, original_channels, attributes):
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.ndim != 1 or original_channels < 1 or np.any(indices < 0)
            or np.any(indices >= original_channels) or np.any(np.diff(indices) <= 0)):
        raise ValueError("Invalid ordered source channel indices")
    spacing = np.asarray(attributes.get("Spacing", []), dtype=float).reshape(-1)
    origin = np.asarray(attributes.get("Origin", []), dtype=float).reshape(-1)
    extent = np.asarray(attributes.get("Extent", []), dtype=float).reshape(-1)
    dx = float(spacing[0]) if spacing.size and np.isfinite(spacing[0]) and spacing[0] > 0 else None
    extent_valid = (extent.size >= 2 and np.isfinite(extent[:2]).all()
                    and np.equal(extent[:2], np.floor(extent[:2])).all()
                    and extent[1] - extent[0] + 1 == original_channels)
    global_indices = indices + int(extent[0]) if extent_valid else None
    positions = (origin[0] + global_indices * dx if global_indices is not None and dx is not None
                 and origin.size and np.isfinite(origin[0]) else None)
    differences = np.diff(indices)
    return {"schema_version": 1, "source_channel_count": int(original_channels),
            "source_channel_indices": indices.tolist(), "selected_channel_count": len(indices),
            "selected_rows_contiguous": bool(np.all(differences == 1)),
            "missing_source_rows_inside_selection": int(np.sum(np.maximum(differences - 1, 0))),
            "largest_source_index_step": int(differences.max()) if differences.size else 0,
            "dx_m": dx, "extent_spatial_matches_shape": bool(extent_valid),
            "source_extent_channel_indices": global_indices.tolist() if global_indices is not None else None,
            "dataset_relative_positions_m": (indices * dx).tolist() if dx is not None else None,
            "metadata_positions_m": positions.tolist() if positions is not None else None,
            "coordinate_assumption": "Origin[0] + (Extent[0] + local source channel index)*Spacing[0], in pipeline spatial units assumed metres; not externally calibrated"}


def map_identifier(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_spatial(block, max_channels, attributes):
    block = np.asarray(block)
    if block.ndim != 2:
        raise ValueError("Expected channels x time block")
    finite = np.isfinite(block).all(axis=1)
    indices = selected_indices(finite, max_channels)
    return block[indices], spatial_metadata(indices, block.shape[0], attributes)
