from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from Experiments.runners.run_baseline_2choice import model_baseline_2choice
from Experiments.runners.run_multihash_random_walk import model_multihash_random_walk
from Experiments.runners.run_multihash_stash_random_walk import model_multihash_stash_random_walk
from Experiments.runners.run_bubble_up_harvard import model_bubble_up_harvard
from Experiments.runners.run_fdhc import model_fdic


# ============================================================
# Bins
# ============================================================

BINS: List[Tuple[float, float]] = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.85),
    (0.90, 0.94),
    (0.95, 0.97),
    (0.97, 0.98),
    (0.98, 0.99),
]


def mids(bins_: List[Tuple[float, float]]) -> np.ndarray:
    return np.array([(lo + hi) / 2.0 for lo, hi in bins_], dtype=float)


def to_series(x: np.ndarray, y: List[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y_arr = np.asarray([np.nan if v is None else float(v) for v in y], dtype=float)
    mask = ~np.isnan(y_arr)
    return x[mask], y_arr[mask]


X_MID_PCT = mids(BINS) * 100.0


@dataclass
class ModelResult:
    name: str
    avg_probes: List[Optional[float]]
    fail_load: float


def model_ours() -> ModelResult:
    avg = [1.052, 1.179, 1.346, 1.260, 1.224, 1.671, 2.140, 3.502, 5.477, 7.822, 18.907, 25.590, 45.889]
    fail_load = 0.9940
    return ModelResult("Our Model", avg, fail_load)

def total_capacity_5_layers(cap_power: int, layers: int = 5) -> int:
    """
    Returns sum_{i=0..layers-1} 2^(cap_power + i)
    Default layers=5.
    """
    return sum(2 ** (cap_power + i) for i in range(layers))


def get_all_models() -> List[ModelResult]:
    fdic_base_power = 14
    active_layers = 5

    total_cap = total_capacity_5_layers(fdic_base_power, layers=active_layers)

    return [
        model_baseline_2choice(total_capacity=total_cap, max_kicks=1000, seed=0, bins=BINS),
        model_multihash_random_walk(total_capacity=total_cap, k=17, max_kicks=1000, seed=0, bins=BINS),
        model_multihash_stash_random_walk(total_capacity=total_cap, k=17, max_kicks=1000, stash_size=8, seed=0, bins=BINS),
        model_bubble_up_harvard(total_capacity=total_cap, d=17, max_total_steps=1000, core_run_limit_mult=8, seed=0, bins=BINS),
        model_fdic(cap_power=fdic_base_power, active_layers=active_layers, high_load_threshold=0.95, seed=0, bins=BINS),
    ]



# ============================================================
# Plotting
# ============================================================

def plot_memory_utilization_high_end(models: List[ModelResult]) -> None:
    hi = [m for m in models if m.fail_load >= 0.98]

    names = [m.name for m in hi]
    loads = np.array([m.fail_load for m in hi], dtype=float)

    plt.figure(figsize=(10, 4))
    x = np.arange(len(names))

    plt.bar(x, loads * 100.0)
    plt.xticks(x, names, rotation=18, ha="right")
    plt.ylabel("Max achieved load (%)")
    plt.title("Memory utilization (zoomed: 99–100%)")
    plt.ylim(98.0, 100.0)

    for i, load in enumerate(loads):
        plt.text(
            i,
            load * 100.0 + 0.01,
            f"{load*100:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_avg_probes_full(models: List[ModelResult]) -> None:
    plt.figure(figsize=(10, 5))

    for m in models:
        x2, y2 = to_series(X_MID_PCT, m.avg_probes)
        plt.plot(x2, y2, marker="o", label=m.name)

    plt.xlabel("Load bin midpoint (%)")
    plt.ylabel("Avg probes per insert")
    plt.title("Avg probes per insert vs load (full range)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_avg_probes_zoom(models: List[ModelResult], zoom_from: float = 95.0) -> None:
    plt.figure(figsize=(10, 5))

    for m in models:
        x2, y2 = to_series(X_MID_PCT, m.avg_probes)
        mask = x2 >= zoom_from
        if np.any(mask):
            plt.plot(x2[mask], y2[mask], marker="o", label=m.name)

    plt.xlabel("Load bin midpoint (%)")
    plt.ylabel("Avg probes per insert")
    plt.title(f"Avg probes per insert (zoom: {zoom_from:.0f}%+)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Entry point
# ============================================================

def run_all_plots() -> None:
    models = get_all_models()

    # Figure 1
    plot_memory_utilization_high_end(models)

    # Figure 2
    plot_avg_probes_full(models)

    # Figure 3
    plot_avg_probes_zoom(models, zoom_from=95.0)


if __name__ == "__main__":
    run_all_plots()
