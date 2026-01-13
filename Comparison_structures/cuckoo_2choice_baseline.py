from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Dict


# ============================================================
# Baseline: 2-choice cuckoo hashing (single-cell buckets)
# - Two hash locations per key
# - Probe metric: UNIQUE cuckoo_hashes(x) computations per insertion
# ============================================================


# ---------------- Hashing (2-choice) ----------------

def _h64(salt: bytes, key_bytes: bytes) -> int:
    m = hashlib.sha256()
    m.update(salt)
    m.update(key_bytes)
    return int.from_bytes(m.digest()[:8], "little", signed=False)


def cuckoo_hashes(key: Any, capacity: int) -> Tuple[int, int]:
    kb = str(key).encode("utf-8", errors="surrogatepass")
    h1 = _h64(b"\xA5", kb) % capacity
    h2 = _h64(b"\x5A", kb) % capacity
    if h2 == h1:
        # Ensure two distinct choices (rare but possible).
        h2 = (h2 + 1) % capacity
    return h1, h2


# ---------------- Baseline Cuckoo Table (single-cell) ----------------

@dataclass
class Cuckoo2Choice:
    capacity: int
    max_kicks: int = 500
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.capacity > 0
        random.seed(self.seed)
        self.table: List[Optional[Any]] = [None] * self.capacity
        self.size: int = 0

    def load(self) -> float:
        return self.size / self.capacity

    def contains(self, key: Any) -> bool:
        i1, i2 = cuckoo_hashes(key, self.capacity)
        return (self.table[i1] == key) or (self.table[i2] == key)

    def insert(self, key: Any) -> Tuple[bool, int]:
        """
        Returns: (success, probes)

        probes counts UNIQUE cuckoo_hashes(x) computations within THIS insertion,
        i.e., how many times we computed cuckoo_hashes(x) for some x
        (cached per key during this insert).

        Failure: exceeded max_kicks evictions.
        """
        if key is None:
            raise ValueError("key cannot be None")

        # Cache: key -> (i1,i2) within this insertion.
        hash_cache: Dict[Any, Tuple[int, int]] = {}
        probes = 0

        def get_hashes(k: Any) -> Tuple[int, int]:
            nonlocal probes
            if k in hash_cache:
                return hash_cache[k]
            probes += 1
            h = cuckoo_hashes(k, self.capacity)
            hash_cache[k] = h
            return h

        i1, i2 = get_hashes(key)

        if self.table[i1] == key or self.table[i2] == key:
            return True, probes

        cur = key
        cur_i = i1

        for _ in range(self.max_kicks + 1):
            if self.table[cur_i] is None:
                self.table[cur_i] = cur
                self.size += 1
                return True, probes

            cur, self.table[cur_i] = self.table[cur_i], cur

            a, b = get_hashes(cur)
            cur_i = b if cur_i == a else a

        return False, probes


# ---------------- Experiment: avg probes per load bin + failure point ----------------

def run_cuckoo_load_experiment(
    capacity: int,
    load_bins: List[Tuple[float, float]],
    max_kicks: int = 500,
    seed: int = 0,
    key_prefix: str = "k",
) -> Dict[str, Any]:
    tbl = Cuckoo2Choice(capacity=capacity, max_kicks=max_kicks, seed=seed)

    bin_attempts = [0] * len(load_bins)
    bin_success = [0] * len(load_bins)

    inserted = 0
    while True:
        key = f"{key_prefix}{inserted}"

        load_before = tbl.load()

        ok, probes = tbl.insert(key)
        if not ok:
            return {
                "capacity": capacity,
                "max_kicks": max_kicks,
                "seed": seed,
                "inserted": inserted,
                "failure_load": load_before,
                "bin_avg_probes": [
                    (bin_attempts[i] / bin_success[i]) if bin_success[i] else None
                    for i in range(len(load_bins))
                ],
                "bin_success": bin_success,
                "bin_attempts": bin_attempts,
                "load_bins": load_bins,
            }

        inserted += 1

        for i, (lo, hi) in enumerate(load_bins):
            if lo <= load_before < hi:
                bin_attempts[i] += probes
                bin_success[i] += 1
                break


# ---------------- Output formatting (paper-style) ----------------

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


def print_summary_final_state(res: Dict[str, Any]) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    def fmt_int(n: int) -> str:
        return f"{n:,}"

    capacity = res["capacity"]
    items = res["inserted"]
    occ = items / capacity if capacity else 0.0

    total_inserts = items
    total_probes = sum(res["bin_attempts"])
    avg_probes = (total_probes / total_inserts) if total_inserts else 0.0

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
    print(f"  Total inserts:          {fmt_int(total_inserts)}")
    print(f"  Total probes:           {fmt_int(total_probes)}")
    print(f"  Avg probes / insert:    {avg_probes:.3f}")
    print("----------------")

    print("Rate by GLOBAL load ranges:")
    for (lo, hi), att, suc in zip(
        res["load_bins"], res["bin_attempts"], res["bin_success"]
    ):
        avg = (att / suc) if suc else None
        avg_s = "N/A" if avg is None else f"{avg:.3f}"
        print(
            f"  {int(lo*100):02d}–{int(hi*100):02d}% | "
            f"inserts={fmt_int(suc)} | probes={fmt_int(att)} | avg={avg_s}"
        )

    print("================================================\n")


# ---------------- Public entry point ----------------

def run_baseline_cuckoo_2choice(
    total_capacity,
    max_kicks: int = 1000,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Convenience runner (call from outside).
    Returns the result dict and prints a paper-style summary.
    """
    if bins is None:
        bins = default_load_bins()

    res = run_cuckoo_load_experiment(
        capacity=total_capacity,
        load_bins=bins,
        max_kicks=max_kicks,
        seed=seed,
    )
    print_summary_final_state(res)
    return res