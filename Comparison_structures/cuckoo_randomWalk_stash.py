from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Dict


# ============================================================
# Baseline: k-Choice Random-Walk Cuckoo with Stash
#
# - Each key has k candidate locations (k-choice hashing)
# - Insertions use a uniform random walk on evictions:
#   at each kick, pick the next position uniformly at random
#   among the k choices of the current key
# - If the kick budget is exceeded, the remaining key is placed
#   into a small stash (bounded auxiliary list)
# - Cost metric: UNIQUE table cells touched per insertion
# ============================================================


# ---------------- Hashing (k-choice) ----------------

def _h64(salt: bytes, key_bytes: bytes) -> int:
    m = hashlib.sha256()
    m.update(salt)
    m.update(key_bytes)
    return int.from_bytes(m.digest()[:8], "little", signed=False)


def k_hashes(key: Any, capacity: int, k: int) -> List[int]:
    """
    Return up to k (mostly distinct) indices for the key.
    Deterministic via double hashing.
    """
    kb = str(key).encode("utf-8", errors="surrogatepass")
    base1 = _h64(b"\xA5", kb)
    base2 = _h64(b"\x5A", kb) | 1  

    idxs: List[int] = []
    seen = set()
    for i in range(1, k + 1):
        idx = (base1 + i * base2) % capacity
        if idx in seen:
            idx = (idx + i) % capacity
        idxs.append(idx)
        seen.add(idx)
    return idxs


# ---------------- Table: k-choice + random-walk + stash ----------------

@dataclass
class CuckooKChoiceRandomWalkStash:
    capacity: int
    k: int = 17
    max_kicks: int = 1000
    stash_size: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.capacity > 0
        assert self.k >= 2
        assert self.max_kicks >= 1
        assert self.stash_size >= 0

        random.seed(self.seed)
        self.table: List[Optional[Any]] = [None] * self.capacity
        self.size: int = 0            # counts table + stash elements
        self.stash: List[Any] = []

    def load(self) -> float:
        return self.size / self.capacity

    def insert(self, key: Any) -> Tuple[bool, int]:
        """
        Returns:
            (success, probes_unique_cells)

        probes_unique_cells counts UNIQUE table indices touched within THIS insertion.
        Stash insert is considered a successful insert (but does not touch table cells).
        """
        if key is None:
            raise ValueError("key cannot be None")

        visited: set[int] = set()             # unique cell touches
        hash_cache: Dict[Any, List[int]] = {} # per-insert hash cache
        probes = 0

        def get_idxs(x: Any) -> List[int]:
            v = hash_cache.get(x)
            if v is None:
                v = k_hashes(x, self.capacity, self.k)
                hash_cache[x] = v
            return v

        def touch(idx: int) -> None:
            nonlocal probes
            if idx not in visited:
                visited.add(idx)
                probes += 1

        # Phase 1: direct placement into any of the k positions
        idxs = get_idxs(key)
        for idx in idxs:
            touch(idx)
            cell = self.table[idx]
            if cell is None:
                self.table[idx] = key
                self.size += 1
                return True, probes
            if cell == key:
                return True, probes  # duplicate (normally not used in experiments)

        # Phase 2: random-walk eviction chain
        cur = key
        cur_idx = random.choice(idxs)

        for _ in range(self.max_kicks):
            touch(cur_idx)

            if self.table[cur_idx] is None:
                self.table[cur_idx] = cur
                self.size += 1
                return True, probes

            evicted = self.table[cur_idx]
            self.table[cur_idx] = cur
            cur = evicted

            choices = get_idxs(cur)
            cur_idx = random.choice(choices)

        # Stash fallback
        if self.stash_size > 0 and len(self.stash) < self.stash_size:
            self.stash.append(cur)
            self.size += 1
            return True, probes

        return False, probes


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


def run_cuckoo_load_experiment(
    capacity: int,
    load_bins: List[Tuple[float, float]],
    k: int,
    max_kicks: int = 1000,
    stash_size: int = 8,
    seed: int = 0,
    key_prefix: str = "k",
) -> Dict[str, Any]:
    tbl = CuckooKChoiceRandomWalkStash(
        capacity=capacity,
        k=k,
        max_kicks=max_kicks,
        stash_size=stash_size,
        seed=seed,
    )

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
                "stash_size": stash_size,
                "seed": seed,
                "inserted": inserted,
                "failure_load": load_before,  
                "stash_final": len(tbl.stash),
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
    print(f"  Stash size:             {res['stash_size']}")
    print(f"  Stash used at end:      {fmt_int(res['stash_final'])}")
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

def run_baseline_kchoice_random_walk_stash(
    total_capacity,
    k: int = 17,
    max_kicks: int = 1000,
    stash_size: int = 8,
    seed: int = 0,
    bins: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    if bins is None:
        bins = default_load_bins()

    res = run_cuckoo_load_experiment(
        capacity=total_capacity,
        load_bins=bins,
        k=k,
        max_kicks=max_kicks,
        stash_size=stash_size,
        seed=seed,
    )
    print_summary_final_state(res)
    return res
