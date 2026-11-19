"""Bounded-memory equivalent of the prospective paired physical summary."""
from collections import defaultdict

import numpy as np

from finetuning.prospective_metrics import PhysicalAccumulator
from finetuning.prospective_spectral import merge_spectral, spectral_result, spectral_statistics
from finetuning.run_lifting_study import macro_scores


class StreamingPhysicalSummary:
    def __init__(self, scales, maps):
        self.scales = scales
        self.maps = maps
        self.metrics = defaultdict(PhysicalAccumulator)
        self.costs = defaultdict(lambda: {"frame_bytes": 0, "cells": 0, "windows": 0})
        self.spectra = defaultdict(dict)

    def update(self, row, raw, reconstructed, frame_bytes):
        if raw.shape != reconstructed.shape or raw.ndim != 2 or frame_bytes < 0:
            raise ValueError("Invalid streaming target/decoded geometry or cost")
        key = row["group_id"], row["drive_id"]
        indices = self.maps[row["channel_map_id"]]["source_channel_indices"]
        if len(indices) != len(raw):
            raise ValueError("Physical channel map differs from acquired rows")
        self.costs[key]["frame_bytes"] += int(frame_bytes)
        self.costs[key]["cells"] += raw.size
        self.costs[key]["windows"] += 1
        for channel, source_channel in enumerate(indices):
            self.metrics[key + (source_channel,)].update(raw[channel], reconstructed[channel])
        if "dt_s" in row:
            merge_spectral(self.spectra[key], spectral_statistics(
                raw, reconstructed, float(row["dt_s"]), self.scales[key[0]]))

    def result(self, setup_bytes):
        if not self.costs:
            raise ValueError("No acquired cells evaluated")
        records = []
        for key, cost in sorted(self.costs.items()):
            channels = [value.result(self.scales[key[0]]) for k, value in self.metrics.items() if k[:2] == key]
            records.append({
                "group_id": key[0], "drive_id": key[1], **cost,
                "nrmse_star_macro_channels": float(np.mean([c["nrmse_star"] for c in channels])),
                "p90_channel_nrmse_star": float(np.quantile([c["nrmse_star"] for c in channels], .9)),
                "correlation_defined_channels": sum(c["correlation_defined"] for c in channels),
                "source_channel_indices": [k[2] for k in self.metrics if k[:2] == key],
                "channels": channels, "spectral": spectral_result(self.spectra[key])})
        cells = sum(c["cells"] for c in self.costs.values())
        frames = sum(c["frame_bytes"] for c in self.costs.values())
        return {"records": records, **macro_scores(records), "cells": cells,
                "windows": sum(c["windows"] for c in self.costs.values()),
                "frame_bytes": frames, "setup_bytes": setup_bytes, "stream_bytes": frames + setup_bytes,
                "stream_bpc": 8 * (frames + setup_bytes) / cells}
