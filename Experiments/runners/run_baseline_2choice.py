
from typing import List, Optional, Tuple
from Experiments.common import BINS, ModelResult
from Comparison_structures.cuckoo_2choice_baseline import (
    run_baseline_cuckoo_2choice,
)



def _avg_for_bins(
    src_bins: List[Tuple[float, float]],
    src_avg: List[Optional[float]],
    dst_bins: List[Tuple[float, float]],
) -> List[Optional[float]]:
    """
    Map avg values from src_bins to dst_bins by exact bin match.
    If a dst bin is missing in src, returns None for that bin.
    """
    m = {b: src_avg[i] for i, b in enumerate(src_bins)}
    return [m.get(b, None) for b in dst_bins]


def model_baseline_2choice(
    total_capacity,
    max_kicks: int = 1000,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,):
    """
    Baseline: 2-choice cuckoo hashing (single-cell buckets).
    Calls the real runner and converts to ModelResult.
    """
    if bins is None:
        bins = BINS  
    res = run_baseline_cuckoo_2choice(
        total_capacity,
        max_kicks=max_kicks,
        seed=seed,
        bins=None,  
    )

    src_bins = res["load_bins"]
    src_avg = res["bin_avg_probes"]
    fail_load = float(res["failure_load"])

    avg = _avg_for_bins(src_bins, src_avg, bins)

    return ModelResult("Baseline", avg, fail_load)
