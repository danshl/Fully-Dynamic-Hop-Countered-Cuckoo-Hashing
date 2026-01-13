from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Dict


# ============================================================
# Baseline: Random-Walk k-Choice Cuckoo Hashing
#
# - Each key has k possible locations (k-choice hashing)
# - Insertions use a TRUE random walk:
#   at each eviction step, the next position is chosen uniformly at random
#   among the k choices of the current key
# - Cost metric: UNIQUE cell probes per insertion
# ============================================================


# ---------------- Hashing (k-choice) ----------------

def _h64(salt: bytes, key_bytes: bytes) -> int:
    m = hashlib.sha256()
    m.update(salt)
    m.update(key_bytes)
    return int.from_bytes(m.digest()[:8], "little", signed=False)


def k_hashes(key: Any, capacity: int, k: int) -> List[int]:
    """
    Return up to k distinct table indices for the key.
    Indices are generated deterministically using double hashing.
    """
    kb = str(key).encode("utf-8", errors="surrogatepass")
    base1 = _h64(b"\xA5", kb)
    base2 = _h64(b"\x5A", kb) | 1  # ensure odd stride

    idxs: List[int] = []
    seen = set()
    for i in range(1, k + 1):
        idx = (base1 + i * base2) % capacity
        if idx in seen:
            idx = (idx + i) % capacity
        idxs.append(idx)
        seen.add(idx)
    return idxs


# ---------------- Cuckoo Table (single-cell, k-choice) ----------------

@dataclass
class CuckooKChoice:
    capacity: int
    k: int = 2
    max_kicks: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.capacity > 0
        assert self.k >= 2
        random.seed(self.seed)
        self.table: List[Optional[Any]] = [None] * self.capacity
        self.size: int = 0

    def load(self) -> float:
        return self.size / self.capacity

    def insert(self, key: Any) -> Tuple[bool, int]:
        """
        Random-walk k-choice cuckoo insertion.

        Returns:
            (success, probes)

        probes = number of UNIQUE table cells touched during THIS insertion
        (each index counted at most once via a per-insert cache).

        Failure = exceeded max_kicks evictions.
        """
        visited: set[int] = set()          # per-insert probe cache
        hash_cache: Dict[Any, List[int]] = {}
        probes = 0

        def get_idxs(x: Any) -> List[int]:
            if x not in hash_cache:
                hash_cache[x] = k_hashes(x, self.capacity, self.k)
            return hash_cache[x]

        def touch(idx: int) -> None:
            nonlocal probes
            if idx not in visited:
                visited.add(idx)
                probes += 1

        # Phase 1: direct placement
        idxs = get_idxs(key)
        for idx in idxs:
            touch(idx)
            if self.table[idx] is None:
                self.table[idx] = key
                self.size += 1
                return True, probes

        # Phase 2: random-walk eviction
        cur = key
        cur_idxs = idxs
        cur_idx = random.choice(cur_idxs)

        for _ in range(self.max_kicks):
            touch(cur_idx)

            if self.table[cur_idx] is None:
                self.table[cur_idx] = cur
                self.size += 1
                return True, probes

            evicted = self.table[cur_idx]
            self.table[cur_idx] = cur
            cur = evicted

            cur_idxs = get_idxs(cur)
            cur_idx = random.choice(cur_idxs)

        return False, probes


# ---------------- Experiment: avg probes per load bin + failure point ----------------

def run_cuckoo_load_experiment(
    capacity: int,
    load_bins: List[Tuple[float, float]],
    k: int,
    max_kicks: int = 1000,
    seed: int = 0,
    key_prefix: str = "k",
) -> Dict[str, Any]:
    tbl = CuckooKChoice(capacity=capacity, k=k, max_kicks=max_kicks, seed=seed)

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
                "k": k,
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

def print_summary_final_state(res: Dict[str, Any]) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    def fmt_int(n: int) -> str:
        return f"{n:,}"

    capacity = res["capacity"]
    items = res["inserted"]
    occ = items / capacity if capacity else 0.0

    total_probes = sum(res["bin_attempts"])
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

def run_baseline_kchoice_random_walk(
    total_capacity,
    k: int = 17,
    max_kicks: int = 1000,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    if bins is None:
        bins = [
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

    res = run_cuckoo_load_experiment(
        capacity=total_capacity,
        load_bins=bins,
        k=k,
        max_kicks=max_kicks,
        seed=seed,
    )
    print_summary_final_state(res)
    return res
