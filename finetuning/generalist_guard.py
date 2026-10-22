"""Experimental fixed-reference admission; not an application acceptance test."""
import math
from statistics import mean


GUARD_METRICS = {"relative_rmse": 1, "correlation": -1, "ordinary_channel_p90_relative_rmse": 1}


def grouped(rows):
    result = {(r["family"], r["configuration"]): r for r in rows}
    if not result or len(result) != len(rows):
        raise ValueError("Empty or duplicate configuration groups")
    return result


def assess(baseline, candidate, epsilon=1e-9):
    before, after = grouped(baseline), grouped(candidate)
    if before.keys() != after.keys():
        raise ValueError("Configuration coverage differs")
    violations = []
    families = {}
    for key, row in after.items():
        if int(row["windows"]) != int(before[key]["windows"]):
            raise ValueError("Configuration window counts differ")
        for metric, direction in GUARD_METRICS.items():
            a, b = float(before[key][metric]), float(row[metric])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite guard metric")
            if direction * (b - a) > epsilon:
                violations.append({"family": key[0], "configuration": key[1], "metric": metric,
                                   "baseline": a, "candidate": b, "signed_regression": direction * (b - a)})
        families.setdefault(key[0], []).append(float(row["relative_rmse"]))
    return {"eligible": not violations, "macro_relative_rmse": mean(mean(v) for v in families.values()),
            "violations": violations}
