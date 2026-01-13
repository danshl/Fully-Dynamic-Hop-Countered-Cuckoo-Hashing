from typing import Dict, Tuple
import matplotlib.pyplot as plt

from FDHCcuckoo import run_single_simulation


def avg_probes_in_range(tbl, load_range: Tuple[float, float]) -> float:
    lo, hi = load_range
    stats = tbl.stats.range_stats.get((lo, hi))
    if not stats or stats["inserts"] == 0:
        return float("nan")
    return stats["probes"] / stats["inserts"]


if __name__ == "__main__":
    base_powers = [11, 13, 15]
    taus = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
    load_range = (0.95, 0.97)
    K = 5
    seed = 0

    results: Dict[int, Dict[float, float]] = {}

    for p in base_powers:
        results[p] = {}
        for tau in taus:
            tbl = run_single_simulation(
                base_power=p,
                active_layers=K,
                high_load_threshold=0.95,      # leave fixed (or choose what you want)
                load_ranges=[load_range],
                seed=seed,
                active_layer_threshold=tau,     # NEW: this is the knob you sweep
            )
            results[p][tau] = avg_probes_in_range(tbl, load_range)

    plt.figure(figsize=(7.8, 4.6))
    for p in sorted(results.keys()):
        y = [results[p][tau] for tau in taus]
        plt.plot(taus, y, marker="o", linewidth=2, label=f"$2^{{{p}}}$")

    plt.xlabel(r"Active-layer threshold $\tau$ (ignore layers with occupancy $>\tau$)")
    plt.ylabel("Average probes per successful insertion\n(global load 95%–97%)")
    plt.xticks(taus)
    plt.grid(True, linewidth=0.6, alpha=0.6)
    plt.legend(title="Base capacity")
    plt.title(r"Sensitivity to active-layer threshold $\tau$")
    plt.tight_layout()
    plt.show()
