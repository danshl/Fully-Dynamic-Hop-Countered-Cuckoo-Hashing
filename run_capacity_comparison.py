from FDHCcuckoo.sim import run_single_simulation

# Load ranges to measure
BINS = [
    (0.80, 0.90),
    (0.90, 0.94),
    (0.95, 0.97),
    (0.97, 0.98),
    (0.98, 0.99),
]

BIN_LABELS = ["80-90%", "90-94%", "95-97%", "97-98%", "98-99%"]

# base_power 15 -> 14 -> 13 -> 12 -> 11  (largest to smallest, matching paper table)
BASE_POWERS = [15, 14, 13, 12, 11]


def total_capacity(base_power: int, layers: int = 5) -> int:
    return sum(2 ** (base_power + i) for i in range(layers))


def fmt_cap(cap: int) -> str:
    if cap >= 1_000_000:
        return f"{cap / 1_000_000:.2f}M"
    return f"{cap:,}"


if __name__ == "__main__":
    results = {}

    for p in BASE_POWERS:
        cap = total_capacity(p)
        print(f"Running base_power={p}  capacity={fmt_cap(cap)} ...")
        tbl = run_single_simulation(
            base_power=p,
            active_layers=5,
            high_load_threshold=0.95,
            load_ranges=BINS,
            seed=0,
        )

        avgs = []
        for b in BINS:
            stats = tbl.stats.range_stats.get(b)
            if stats and stats["inserts"] > 0:
                avgs.append(stats["probes"] / stats["inserts"])
            else:
                avgs.append(None)

        results[cap] = avgs

    # ── print results ──────────────────────────────────────────
    caps = [total_capacity(p) for p in BASE_POWERS]
    col_w = 14

    print("\n" + "=" * (10 + col_w * len(caps)))
    header = f"{'Load':>10}" + "".join(f"{fmt_cap(c):>{col_w}}" for c in caps)
    print(header)
    print("-" * (10 + col_w * len(caps)))

    for i, label in enumerate(BIN_LABELS):
        row = f"{label:>10}"
        for cap in caps:
            val = results[cap][i]
            row += f"{'N/A':>{col_w}}" if val is None else f"{val:>{col_w}.2f}"
        print(row)

    print("=" * (10 + col_w * len(caps)))
