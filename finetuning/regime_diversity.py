"""Metadata-only nested regime subsets and equal-cell training crop policy."""
from collections import defaultdict

import numpy as np


def nested_groups(rows):
    families = defaultdict(set)
    for row in rows:
        families[row["group_id"].split("/")[0]].add(row["group_id"])
    if set(families) != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Expected four FEBUS training families")
    if sum(map(len, families.values())) != 10:
        raise ValueError("Expected ten training configurations")
    ordered = {family: sorted(groups) for family, groups in families.items()}
    small = {groups[0] for groups in ordered.values()}
    medium = small | {ordered["multicore"][1], ordered["single_mode"][1]}
    return {4: sorted(small), 6: sorted(medium),
            10: sorted(group for groups in ordered.values() for group in groups)}


def crop(raw, rng):
    if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[0] < 8 or raw.shape[1] < 1000:
        raise ValueError("Expected eligible float32 8x1000 training carrier")
    offset = int(rng.integers(raw.shape[0] - 8 + 1))
    start = int(rng.integers(raw.shape[1] - 1000 + 1))
    return raw[offset:offset + 8, start:start + 1000].copy(), offset, start
