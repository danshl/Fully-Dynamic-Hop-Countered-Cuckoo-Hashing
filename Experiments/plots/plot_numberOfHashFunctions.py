from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

from FDHCcuckoo import run_single_simulation


# ------------------------------------------------------------
# Data extraction (directly from table state)
# ------------------------------------------------------------
def extract_layer_level_stats(tbl) -> Dict[str, Any]:
    """
    Extract per-layer placement-index statistics directly from the table state.
    """
    snapshot: Dict[str, Any] = {
        "base_capacity": tbl.base_capacity,
        "num_items": int(tbl.num_items),
        "global_load": float(tbl.global_load_factor()),
        "layers": [],
    }

    for layer_id, L in enumerate(tbl.layers):
        # placement index (level) -> number of elements
        level_counts = {int(lvl): int(len(keys)) for lvl, keys in L.iter_levels_desc()}

        snapshot["layers"].append({
            "layer_id": int(layer_id),
            "size": int(L.size),
            "capacity": int(L.capacity),
            "occupancy": float(L.occupancy()),
            "max_i_used": int(L.max_hash_index_used),
            "levels": level_counts,
        })

    return snapshot


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_dist(level_counts: Dict[int, int]) -> Dict[int, int]:
    """
    Fill missing levels with 0 up to max level, so lines look consistent.
    """
    if not level_counts:
        return {}
    m = max(level_counts.keys())
    return {lvl: int(level_counts.get(lvl, 0)) for lvl in range(1, m + 1)}


def dist_to_cdf(level_counts: Dict[int, int]) -> Tuple[List[int], List[float]]:
    """
    Convert {level -> count} into CDF arrays (xs=levels, ys=cdf in [0,1]).
    """
    if not level_counts:
        return [], []

    level_counts = _normalize_dist(level_counts)
    levels = sorted(level_counts.keys())
    total = sum(level_counts.values())
    if total == 0:
        return [], []

    xs: List[int] = []
    ys: List[float] = []
    cum = 0

    for lvl in levels:
        cum += level_counts[lvl]
        xs.append(lvl)
        ys.append(cum / total)

    return xs, ys


def merge_all_layers(stats: Dict[str, Any]) -> Dict[int, int]:
    """
    Merge placement-index distributions from all layers into one global dist.
    """
    merged: Dict[int, int] = {}
    for layer in stats["layers"]:
        for lvl, cnt in layer["levels"].items():
            merged[int(lvl)] = merged.get(int(lvl), 0) + int(cnt)
    return merged


def get_last_layer_dist(stats: Dict[str, Any]) -> Dict[int, int]:
    """
    Get levels distribution for the last (largest) layer.
    """
    last_layer = max(stats["layers"], key=lambda x: x["layer_id"])
    return last_layer["levels"]


# ------------------------------------------------------------
# Main experiment: base_power = 11..15 inclusive + plots
# ------------------------------------------------------------
if __name__ == "__main__":
    BASE_POWERS = range(11, 16)  # 11..15 inclusive

    stats_by_power: Dict[int, Dict[str, Any]] = {}

    # Run simulations
    for p in BASE_POWERS:
        tbl = run_single_simulation(
            base_power=p,
            active_layers=5,
            high_load_threshold=0.95,
            seed=0,
        )
        stats_by_power[p] = extract_layer_level_stats(tbl)

    # --------------------------------------------------------
    # Plot 1: CDF of placement index – last layer only
    # --------------------------------------------------------
    plt.figure(figsize=(8, 4.8))
    for p in sorted(stats_by_power.keys()):
        last_dist = get_last_layer_dist(stats_by_power[p])
        xs, ys = dist_to_cdf(last_dist)
        plt.plot(xs, [v * 100 for v in ys], marker="o", linewidth=2, label=f"2^{p}")

    plt.xlabel("Placement index d(x)")
    plt.ylabel("CDF: P(d(x) ≤ t) [%]")
    plt.title("CDF of placement index (last layer)")
    plt.grid(True, alpha=0.6)
    plt.legend(title="Base capacity")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 2: Global CDF – all layers combined
    # --------------------------------------------------------
    plt.figure(figsize=(8, 4.8))
    for p in sorted(stats_by_power.keys()):
        global_dist = merge_all_layers(stats_by_power[p])
        xs, ys = dist_to_cdf(global_dist)
        plt.plot(xs, [v * 100 for v in ys], marker="o", linewidth=2, label=f"2^{p}")

    plt.xlabel("Placement index d(x)")
    plt.ylabel("CDF: P(d(x) ≤ t) [%]")
    plt.title("Global CDF of placement index (all layers)")
    plt.grid(True, alpha=0.6)
    plt.legend(title="Base capacity")
    plt.tight_layout()
    plt.show()
