from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Dict


# ============================================================
# Baseline: Basic Bubble-Up (Section 4)
#
# - Single table (one layer), single-cell buckets
# - d-ary hashing: each key has d candidate positions h1..hd
# - Insertion follows the "basic bubble-up" move rules (Type 1..4)
# - Cost metric: UNIQUE table cells touched per insertion
# - Stops on either:
#   (1) too many consecutive Type1/Type2 moves (core run limit), or
#   (2) max_total_steps (explicit loop budget)
# ============================================================


# ---------------- Hashing: d positions h1..hd ----------------

def _h64(salt: bytes, key_bytes: bytes) -> int:
    m = hashlib.sha256()
    m.update(salt)
    m.update(key_bytes)
    return int.from_bytes(m.digest()[:8], "little", signed=False)


def d_hashes(key: Any, capacity: int, d: int) -> List[int]:
    """
    Deterministic d-ary hashing using double hashing.
    Returns indices for h1..hd (list is 0-based; paper uses 1-based).
    """
    kb = str(key).encode("utf-8", errors="surrogatepass")
    base1 = _h64(b"\xA5", kb)
    base2 = _h64(b"\x5A", kb) | 1  # odd step

    idxs: List[int] = []
    seen = set()
    for i in range(1, d + 1):
        idx = (base1 + i * base2) % capacity
        if idx in seen:
            idx = (idx + i) % capacity
        idxs.append(idx)
        seen.add(idx)
    return idxs


# ---------------- Basic Bubble-Up Table (single-cell) ----------------

@dataclass
class BasicBubbleUp:
    capacity: int
    d: int
    core_run_limit_mult: int = 8
    max_total_steps: int = 1000

    def __post_init__(self) -> None:
        assert self.capacity > 0
        assert self.d >= 3
        assert self.core_run_limit_mult >= 1
        assert self.max_total_steps >= 1

        self.table: List[Optional[Any]] = [None] * self.capacity
        self.size: int = 0

        # Approximates the "omega(log n)" stop criterion with const*log2(n)
        self.core_run_limit = max(
            1, self.core_run_limit_mult * int(math.log2(max(2, self.capacity)))
        )

    def load(self) -> float:
        return self.size / self.capacity

    def insert(self, key: Any) -> Tuple[bool, int, int]:
        """
        Returns:
            (success, probes_unique_cells, steps)

        probes_unique_cells counts UNIQUE table indices touched within THIS insertion.
        steps counts loop iterations (move attempts).
        """
        if key is None:
            raise ValueError("key cannot be None")

        visited: set[int] = set()              # unique cell touches
        hash_cache: Dict[Any, List[int]] = {}  # element -> its d indices (per-insert)
        probes = 0
        steps = 0

        def get_idxs(x: Any) -> List[int]:
            v = hash_cache.get(x)
            if v is None:
                v = d_hashes(x, self.capacity, self.d)
                hash_cache[x] = v
            return v

        def touch(idx: int) -> None:
            nonlocal probes
            if idx not in visited:
                visited.add(idx)
                probes += 1

        def choice_from_pos(x: Any, pos: int) -> int:
            """
            choice(x) = i such that h_i(x) == pos, where i in [1..d].
            """
            idxs = get_idxs(x)
            for i, p in enumerate(idxs, start=1):
                if p == pos:
                    return i
            return 1  # fallback (rare)

        x = key
        choice = 0  # new element starts unplaced: choice(x)=0

        consecutive_type12 = 0

        while True:
            steps += 1
            if steps > self.max_total_steps:
                return False, probes, steps

            idxs = get_idxs(x)

            # Type 1: if choice(x)=d -> move to h_{d-1}(x)
            if choice == self.d:
                target = idxs[self.d - 2]  # h_{d-1}
                consecutive_type12 += 1

            # Type 2: if choice(x)=d-1 -> move to h_d(x)
            elif choice == self.d - 1:
                target = idxs[self.d - 1]  # h_d
                consecutive_type12 += 1

            # Type 3: if choice(x)<d-2 -> scan h_{choice+1}..h_{d-2} for empty
            elif choice < self.d - 2:
                consecutive_type12 = 0

                found_free: Optional[int] = None
                for i in range(choice + 1, self.d - 1):  # up to d-2 inclusive (1-based)
                    pos = idxs[i - 1]
                    touch(pos)
                    if self.table[pos] is None:
                        found_free = pos
                        break

                if found_free is not None:
                    self.table[found_free] = x
                    self.size += 1
                    return True, probes, steps

                # Type 4 (fallback after scan): move to h_{d-1}(x)
                target = idxs[self.d - 2]

            # Type 4: if choice(x)=d-2 -> move to h_{d-1}(x)
            else:
                consecutive_type12 = 0
                target = idxs[self.d - 2]

            if consecutive_type12 > self.core_run_limit:
                return False, probes, steps

            # Execute move to target: place x; evict if occupied
            touch(target)
            if self.table[target] is None:
                self.table[target] = x
                self.size += 1
                return True, probes, steps

            evicted = self.table[target]
            self.table[target] = x

            x = evicted
            choice = choice_from_pos(x, target)


# ---------------- Experiment ----------------

def default_load_bins() -> List[Tuple[float, float]]:
    return [
        (0.00, 0.10),
        (0.10, 0.20),
        (0.20, 0.30),
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 0.92),
        (0.92, 0.94),
        (0.94, 0.95),
        (0.95, 0.96),
        (0.96, 0.97),
        (0.97, 0.98),
        (0.98, 0.99),
    ]


def run_bubbleup_load_experiment(
    capacity: int,
    d: int,
    load_bins: List[Tuple[float, float]],
    max_total_steps: int = 1000,
    core_run_limit_mult: int = 8,
    key_prefix: str = "k",
) -> Dict[str, Any]:
    tbl = BasicBubbleUp(
        capacity=capacity,
        d=d,
        max_total_steps=max_total_steps,
        core_run_limit_mult=core_run_limit_mult,
    )

    bin_probes_sum = [0] * len(load_bins)
    bin_success = [0] * len(load_bins)

    inserted = 0
    while True:
        key = f"{key_prefix}{inserted}"
        load_before = tbl.load()

        ok, probes, _steps = tbl.insert(key)
        if not ok:
            return {
                "capacity": capacity,
                "d": d,
                "max_total_steps": max_total_steps,
                "core_run_limit_mult": core_run_limit_mult,
                "core_run_limit": tbl.core_run_limit,
                "inserted": inserted,
                "failure_load": load_before,
                "bin_avg_probes": [
                    (bin_probes_sum[i] / bin_success[i]) if bin_success[i] else None
                    for i in range(len(load_bins))
                ],
                "bin_success": bin_success,
                "bin_probes_sum": bin_probes_sum,
                "load_bins": load_bins,
            }

        inserted += 1

        # Attribute probes to load BEFORE insertion.
        for i, (lo, hi) in enumerate(load_bins):
            if lo <= load_before < hi:
                bin_probes_sum[i] += probes
                bin_success[i] += 1
                break


# ---------------- Output formatting (paper-style) ----------------

def print_summary_final_state(res: Dict[str, Any]) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    def fmt_int(n: int) -> str:
        return f"{n:,}"

    capacity = res["capacity"]
    items = res["inserted"]
    occ = items / capacity if capacity else 0.0

    total_probes = sum(res["bin_probes_sum"])
    avg_probes = (total_probes / items) if items else 0.0

    print("\n================= FINAL STATE =================")
    print(f"Layers: 1   Base capacity: {fmt_int(capacity)}")
    print(
        f"Global: items={fmt_int(items)} / capacity={fmt_int(capacity)}   "
        f"occupancy={pct(occ)}"
    )
    print("------------------------------------------------")
    print(f"{'L#':>3} | {'capacity':>12} | {'size':>12} | {'occ':>6} | {'max_i':>5}")
    print("-" * 70)
    print(
        f"{0:>3} | {fmt_int(capacity):>12} | {fmt_int(items):>12} | "
        f"{pct(occ):>6} | {'-':>5}"
    )
    print("-" * 70)
    print("Status:")
    print(f"  Total inserts:          {fmt_int(items)}")
    print(f"  Total probes:           {fmt_int(total_probes)}")
    print(f"  Avg probes / insert:    {avg_probes:.3f}")
    print("----------------")
    print(f"  d:                      {res['d']}")
    print(f"  core_run_limit_mult:    {res['core_run_limit_mult']}")
    print(f"  core_run_limit:         {res['core_run_limit']}")
    print(f"  max_total_steps:        {res['max_total_steps']}")
    print("----------------")
    print("Rate by GLOBAL load ranges:")
    for (lo, hi), att, suc in zip(
        res["load_bins"], res["bin_probes_sum"], res["bin_success"]
    ):
        avg = (att / suc) if suc else None
        avg_s = "N/A" if avg is None else f"{avg:.3f}"
        print(
            f"  {int(lo*100):02d}–{int(hi*100):02d}% | "
            f"inserts={fmt_int(suc)} | probes={fmt_int(att)} | avg={avg_s}"
        )
    print("================================================\n")


# ---------------- Public entry point ----------------

def run_baseline_bubble_up_basic(
    cap_power: int = 18,
    d: int = 17,
    max_total_steps: int = 1000,
    core_run_limit_mult: int = 8,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    # Bubble-Up baseline here is deterministic given the key sequence and parameters.
    # (seed is kept for interface symmetry with other baselines.)
    _ = seed

    if bins is None:
        bins = default_load_bins()

    cap = 2 ** cap_power
    res = run_bubbleup_load_experiment(
        capacity=cap,
        d=d,
        load_bins=bins,
        max_total_steps=max_total_steps,
        core_run_limit_mult=core_run_limit_mult,
        key_prefix="k",
    )
    print_summary_final_state(res)
    return res