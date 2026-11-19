"""Budget-extension gate for H2; not an acceptance or operational-fidelity gate."""
import math


def extension_gate(initial, previous, endpoint, selected, independent, anchor_perplexity, innovation_perplexity):
    values = [initial["macro_families"], previous["macro_families"], endpoint["macro_families"],
              selected["macro_families"], independent["macro_families"],
              anchor_perplexity, innovation_perplexity]
    finite = all(value is not None and math.isfinite(value) and value > 0 for value in values)
    if not finite:
        return {"eligible": False, "reason": "Invalid/nonpositive health or score", "final_acceptance": False}
    checks = {
        "learning_one_percent": selected["macro_families"] <= .99 * initial["macro_families"],
        "last_block_learning_point_two_percent": endpoint["macro_families"] <= .998 * previous["macro_families"],
        "within_three_percent_of_independent": selected["macro_families"] <= 1.03 * independent["macro_families"],
        "within_five_percent_of_independent_bytes": selected["stream_bytes"] <= 1.05 * independent["stream_bytes"],
        "both_books_healthy": min(anchor_perplexity, innovation_perplexity) >= 16,
        "same_check_cells": selected["cells"] == independent["cells"],
    }
    return {"eligible": all(checks.values()), "checks": checks,
            "final_acceptance": False, "meaning": "Permission to spend the registered remaining budget only"}
