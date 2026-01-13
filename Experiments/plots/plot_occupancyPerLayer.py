from typing import Dict, List
import matplotlib.pyplot as plt

from FDHCcuckoo import run_single_simulation


# ------------------------------------------------------------
# Helper: F from base_power (d) and K active layers
# ------------------------------------------------------------
def F_from_base_power(base_power: int, K: int) -> int:
    return (1 << base_power) * ((1 << K) - 1)


# ------------------------------------------------------------
# Main: run base_power
# ------------------------------------------------------------
if __name__ == "__main__":
    K = 5
    base_powers = range(11, 16)

    plt.figure(figsize=(8, 4.5))
    x = list(range(K))

    for p in base_powers:
        tbl = run_single_simulation(
            base_power=p,
            active_layers=K,
            high_load_threshold=0.95,
            seed=0,
        )

        layers = tbl.layers[-K:] if len(tbl.layers) >= K else tbl.layers
        occs: List[float] = [L.occupancy() for L in layers]

        if len(occs) < K:
            occs = ([0.0] * (K - len(occs))) + occs

        y = [v * 100 for v in occs]
        F = F_from_base_power(p, K)

        plt.plot(x, y, marker="o", linewidth=2, label=f"F={F:,}")

    plt.xticks(x, [f"L{i}" for i in x])
    plt.ylabel("Occupancy (%)")
    plt.xlabel("Layer")
    plt.ylim(98.5, 100.0)
    plt.grid(True, linewidth=0.6, alpha=0.6)
    plt.legend(title="Total capacity F", ncols=2)
    plt.title("Per-layer occupancy at saturation")
    plt.tight_layout()
    plt.show()
