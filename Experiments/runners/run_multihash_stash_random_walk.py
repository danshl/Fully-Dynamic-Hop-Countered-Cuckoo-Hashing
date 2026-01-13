from __future__ import annotations

from typing import List, Optional, Tuple

from Experiments.common import BINS, ModelResult
from Comparison_structures.cuckoo_randomWalk_stash import (
    run_baseline_kchoice_random_walk_stash,
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


def model_multihash_stash_random_walk(
    total_capacity,
    k: int = 17,
    max_kicks: int = 1000,
    stash_size: int = 8,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> ModelResult:
    """
    Multi-hash + stash + rand (k-choice random-walk cuckoo with stash).
    Calls the real runner and converts to Experiments.common.ModelResult.
    """
    if bins is None:
        bins = BINS

    res = run_baseline_kchoice_random_walk_stash(
        total_capacity,
        k=k,
        max_kicks=max_kicks,
        stash_size=stash_size,
        seed=seed,
        bins=bins,  # use runner's default bins, then map to plot bins
    )

    src_bins = res["load_bins"]
    src_avg = res["bin_avg_probes"]
    fail_load = float(res["failure_load"])

    avg = _avg_for_bins(src_bins, src_avg, bins)

    return ModelResult("Multi-hash + stash + rand", avg, fail_load)
