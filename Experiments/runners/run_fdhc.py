from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from Experiments.common import BINS, ModelResult

from FDHCcuckoo.config import CuckooConfig
from FDHCcuckoo.table import MultiLayerCuckoo


# ============================================================
# FDIC runner 
# ============================================================

def run_fdic_simulation(
    cap_power: int = 23,
    active_layers: int = 5,
    high_load_threshold: float = 0.95,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
    key_prefix: str = "k",
    print_final_state: bool = True,
) -> Dict[str, Any]:
    """
    Runs FDIC (MultiLayerCuckoo) until insert_or_open fails.
    Collects avg probes per bin using tbl.stats.range_stats (load_ranges),
    and returns:
      - load_bins
      - bin_avg_probes
      - failure_load (load BEFORE first failing insert)
    """

    if bins is None:
        bins = BINS

    random.seed(seed)

    cfg = CuckooConfig(
        active_layers=active_layers,
        high_load_threshold=high_load_threshold,
        load_ranges=bins,
    )

    tbl = MultiLayerCuckoo(base_capacity=2 ** cap_power, cfg=cfg)

    inserted = 0
    failure_load: float = 0.0

    while True:
        key = f"{key_prefix}{inserted}"

        load_before = tbl.global_load_factor()

        ok = tbl.insert_or_open(key)
        if not ok:
            failure_load = load_before
            break

        inserted += 1

    bin_avg: List[Optional[float]] = []
    for b in bins:
        st = tbl.stats.range_stats.get(b)
        if not st or st["inserts"] == 0:
            bin_avg.append(None)
        else:
            bin_avg.append(st["probes"] / st["inserts"])

    res = {
        "cap_power": cap_power,
        "capacity": 2 ** cap_power,
        "active_layers": active_layers,
        "high_load_threshold": high_load_threshold,
        "seed": seed,
        "inserted": inserted,
        "failure_load": failure_load,
        "load_bins": bins,
        "bin_avg_probes": bin_avg,
    }

    if print_final_state:
        tbl.print_loads()

    return res


# ============================================================
# Adapter for plots: returns ModelResult
# ============================================================

def model_fdic(
    cap_power: int = 23,
    active_layers: int = 5,
    high_load_threshold: float = 0.95,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> ModelResult:
    """
    FDIC model for plotting.
    Runs the real simulation and converts output -> ModelResult.
    """
    if bins is None:
        bins = BINS

    res = run_fdic_simulation(
        cap_power=cap_power,
        active_layers=active_layers,
        high_load_threshold=high_load_threshold,
        seed=seed,
        bins=bins,
        print_final_state=True,
    )

    avg = res["bin_avg_probes"]
    fail_load = float(res["failure_load"])

    return ModelResult("FDIC", avg, fail_load)


if __name__ == "__main__":
    model_fdic(cap_power=23, active_layers=5, high_load_threshold=0.95, seed=0, bins=BINS)
