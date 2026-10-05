"""Training-only sampling from the prepared manifest; never moves split boundaries."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


def sample_groups(paths: list[Path], manifest: Path) -> dict[tuple[str, str], list[Path]]:
    with manifest.open(newline="") as stream:
        metadata = {Path(row["sample_path"]).resolve(): row for row in csv.DictReader(stream)}
    groups = defaultdict(list)
    for path in paths:
        row = metadata[path.resolve()]
        if row["split"] != "train":
            raise ValueError(f"Non-training window in sampler: {path}")
        groups[(row["family"], row["configuration"])].append(path)
    return dict(groups)


def epoch_paths(paths: list[Path], groups: dict, mode: str, steps: int, rng=random) -> list[Path]:
    """Balanced modes draw with replacement. An epoch defaults to N train updates."""
    steps = steps or len(paths)
    if steps < 1 or not paths:
        raise ValueError("An epoch needs at least one training window")
    if mode == "window":
        result = []
        while len(result) < steps:
            batch = list(paths)
            rng.shuffle(batch)
            result.extend(batch)
        return result[:steps]
    keys = sorted(groups)
    if mode == "group":
        return [rng.choice(groups[rng.choice(keys)]) for _ in range(steps)]
    if mode != "family-group":
        raise ValueError(f"Unknown sampling mode: {mode}")
    families = sorted({key[0] for key in keys})
    by_family = {family: [key for key in keys if key[0] == family] for family in families}
    return [rng.choice(groups[rng.choice(by_family[rng.choice(families)])]) for _ in range(steps)]
