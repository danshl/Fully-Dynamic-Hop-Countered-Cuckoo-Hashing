from dataclasses import dataclass
import hashlib
import random
from typing import NamedTuple, Optional, Any, List, Tuple, Dict

random.seed(0)
OCC_STOP: float = 1

Active_layers = 5
GSA_best = True
MAINTENANCE_PERIOD = 100

# ---- ranges you want to measure (GLOBAL load factor after each successful insert) ----
LOAD_RANGES: List[Tuple[float, float]] = [
    (0.90, 0.92),
    (0.92, 0.94),
    (0.94, 0.95),
    (0.95, 0.96),
    (0.96, 0.97),
    (0.97, 0.98),
    (0.98, 0.99),
]

# -------- Class ----------
@dataclass
class Cell:
    key: Optional[Any] = None
    placement_level: Optional[int] = None

    @property
    def is_empty(self) -> bool:
        return self.key is None


class HashFamily:
    @staticmethod
    def h(layer_id: int, level: int, key_bytes: bytes, capacity: int) -> int:
        """
        Return a slot index for a given layer, hash function index (level >= 1), and key.
        Uses per-layer salt and double hashing for better distribution.
        """
        assert level >= 1, "hash index level must be >= 1"

        # --- base hashes ---
        m1 = hashlib.sha256()
        m1.update(b'\xA5')                      # salt A
        m1.update(layer_id.to_bytes(4, "little", signed=False))
        m1.update(key_bytes)
        h1 = int.from_bytes(m1.digest()[:8], "little", signed=False)

        m2 = hashlib.sha256()
        m2.update(b'\x5A')                      # salt B
        m2.update(layer_id.to_bytes(4, "little", signed=False))
        m2.update(key_bytes)
        h2 = int.from_bytes(m2.digest()[:8], "little", signed=False) | 1  # make sure h2 is odd

        # --- double hashing formula ---
        idx = (h1 + level * h2) % capacity
        return idx


class GSACandidate(NamedTuple):
    layer_id: int  # absolute layer index in self.layers
    key: Any  # resident key currently in the cell
    resident_level: int  # the placement level (i) the resident was inserted with
    seen_level: int  # the hash level at which we saw this slot during this insert


class Layer:
    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self.capacity = capacity
        self.cells: List[Cell] = [Cell() for _ in range(capacity)]
        self.size: int = 0
        self.max_hash_index_used: int = 0
        self.level_index: dict[int, set[Any]] = {}

    def at(self, index: int) -> Cell:
        return self.cells[index]

    def occupancy(self) -> float:
        return self.size / self.capacity

    def note_level_used(self, level: int) -> None:
        if level > self.max_hash_index_used:
            self.max_hash_index_used = level

    def _recompute_max_level(self) -> None:
        self.max_hash_index_used = max(self.level_index.keys(), default=0)

    def register_cell_level(self, key: Any, level: int | None) -> None:
        if level is None or level < 1:
            return
        bucket = self.level_index.setdefault(level, set())
        bucket.add(key)
        self.note_level_used(level)

    def unregister_cell_level(self, key: Any, level: int | None) -> None:
        if not level:
            return
        s = self.level_index.get(level)
        if s:
            s.discard(key)
            if not s:
                self.level_index.pop(level, None)
                if level == self.max_hash_index_used:
                    self._recompute_max_level()

    def move_cell_level(self, key: Any, old_level: int | None, new_level: int | None) -> None:
        self.unregister_cell_level(key, old_level)
        self.register_cell_level(key, new_level)

    def iter_levels_desc(self):
        for level in sorted(self.level_index.keys(), reverse=True):
            s = self.level_index[level]
            if s:
                yield level, s


class MultiLayerCuckoo:
    def __init__(self, base_capacity: int = 128) -> None:
        assert base_capacity > 0
        self.base_capacity = base_capacity
        self.hashes = HashFamily()
        self.layers: List[Layer] = [Layer(base_capacity)]
        self.num_items: int = 0

        self.stats_total_attempts: int = 0
        self.stats_success_inserts: int = 0

        self.stats_highload_attempts: int = 0
        self.stats_highload_success: int = 0

        # per-load-range stats
        self.range_stats: Dict[Tuple[float, float], Dict[str, int]] = {
            (lo, hi): {"attempts": 0, "success": 0} for (lo, hi) in LOAD_RANGES
        }

    def effective_min_gap(self) -> int:
        lf = self._active_load_factor()
        if lf < 0.92:
            return 3
        return 8

    def contains(self, key: Any) -> bool:
        kb = self._key_to_bytes(key)
        for layer_id, L in enumerate(self.layers):
            if L.size == 0:
                continue
            max_i = max(L.max_hash_index_used, 1)
            cap = L.capacity
            for level in range(1, max_i + 1):
                idx = self.hashes.h(layer_id, level, kb, cap)
                cell = L.at(idx)
                if not cell.is_empty and cell.key == key:
                    return True
        return False

    def _active_load_factor(self, k: int = Active_layers) -> float:
        layers = self.layers[-k:] if len(self.layers) >= k else self.layers
        cap = sum(L.capacity for L in layers) or 0
        sz = sum(L.size for L in layers)
        return (sz / cap) if cap else 0.0

    def _note_ranges_after_insert(self, probes_this_insert: int) -> None:
        """Update per-range stats using GLOBAL load factor AFTER successful insert."""
        occ_after = self._global_load_factor()
        for (lo, hi), bucket in self.range_stats.items():
            if lo <= occ_after <= hi:
                bucket["attempts"] += probes_this_insert
                bucket["success"] += 1

    def insert_with_pending(self, key: Any, active_layers: int = Active_layers) -> tuple[bool, Optional[Any]]:
        assert key is not None

        current_key = key
        key_bytes_current = self._key_to_bytes(current_key)

        gsa_candidate: Optional[GSACandidate] = None
        PROBE_LIMIT = 1000
        probes_this_insert = 0

        while probes_this_insert < PROBE_LIMIT:
            hash_fn_index = 1
            while True:
                eff_gap = self.effective_min_gap()
                if self._global_load_factor() >= OCC_STOP:
                    return False, current_key

                # Choose active (top-most) layers
                if len(self.layers) > active_layers:
                    candidate_layers = self.layers[-active_layers:]
                    layer_offset = len(self.layers) - active_layers
                else:
                    candidate_layers = self.layers
                    layer_offset = 0

                active_layer_pairs = [(layer_offset + i, L) for i, L in enumerate(candidate_layers)]
                filtered_pairs = [(lid, L) for (lid, L) in active_layer_pairs if L.occupancy() < 0.998]
                if filtered_pairs:
                    active_layer_pairs = filtered_pairs

                # --- Index cache per (layer_id, hash_fn_index) ---
                index_cache: dict[tuple[int, int], int] = {}

                def compute_index(layer_id: int, L: Layer) -> Optional[int]:
                    """
                    Compute the slot index for the current (layer_id, hash_fn_index, key_bytes_current).
                    Count a probe only when computing a fresh hash (not when served from cache).
                    """
                    nonlocal probes_this_insert
                    cache_key = (layer_id, hash_fn_index)
                    if cache_key in index_cache:
                        return index_cache[cache_key]

                    idx = self.hashes.h(layer_id, hash_fn_index, key_bytes_current, L.capacity)
                    probes_this_insert += 1
                    if probes_this_insert >= PROBE_LIMIT:
                        self.layer_reduction(max_layers=active_layers)
                        self.jump_reduction(max_layers=active_layers)
                        return None
                    index_cache[cache_key] = idx
                    return idx

                # ---------- A) Try empty slot (top-first) ----------
                aborted = False
                for layer_id, L in reversed(active_layer_pairs):
                    slot_index = compute_index(layer_id, L)
                    if slot_index is None:
                        aborted = True
                        break
                    cell = L.at(slot_index)
                    if cell.is_empty:
                        # Place key; this is the successful insert (size increases here)
                        cell.key = current_key
                        cell.placement_level = hash_fn_index
                        L.size += 1
                        self.num_items += 1
                        L.register_cell_level(current_key, hash_fn_index)

                        # Aggregate stats
                        self.stats_total_attempts += probes_this_insert
                        self.stats_success_inserts += 1

                        # per-range stats (GLOBAL load after insert)
                        self._note_ranges_after_insert(probes_this_insert)

                        # high-load stats (ACTIVE load factor)
                        if self._active_load_factor(active_layers) >= HIGH_LOAD_THRESHOLD:
                            self.stats_highload_attempts += probes_this_insert
                            self.stats_highload_success += 1

                        # Periodic maintenance
                        if self.stats_success_inserts % MAINTENANCE_PERIOD == 0:
                            self.layer_reduction(max_layers=active_layers)
                            self.jump_reduction(max_layers=active_layers)

                        return True, None

                if aborted:
                    return False, current_key

                # ---------- B) Try swap (evict resident if allowed) ----------
                did_swap = False
                for layer_id, L in active_layer_pairs:
                    slot_index_current = compute_index(layer_id, L)
                    if slot_index_current is None:
                        return False, current_key

                    cell = L.at(slot_index_current)
                    if not cell.is_empty:
                        resident_level = cell.placement_level
                        if hash_fn_index >= (resident_level + eff_gap):
                            evicted_key = cell.key
                            evicted_level = cell.placement_level
                            L.unregister_cell_level(evicted_key, evicted_level)

                            cell.key = current_key
                            cell.placement_level = hash_fn_index
                            L.register_cell_level(current_key, hash_fn_index)

                            current_key = evicted_key
                            key_bytes_current = self._key_to_bytes(current_key)
                            gsa_candidate = None
                            did_swap = True
                            break
                        else:
                            kb_resident = self._key_to_bytes(cell.key)
                            slot_index_resident = self.hashes.h(layer_id, resident_level, kb_resident, L.capacity)
                            if slot_index_current != slot_index_resident:
                                raise Exception(
                                    f"[SANITY] resident misaligned: L{layer_id} "
                                    f"current_idx={slot_index_current} != resident_idx={slot_index_resident} "
                                    f"for key={cell.key!r}@level={resident_level}"
                                )
                            if (gsa_candidate is None) or (resident_level < gsa_candidate.resident_level):
                                gsa_candidate = GSACandidate(layer_id, cell.key, resident_level, hash_fn_index)

                if did_swap:
                    break

                # ---------- C) GSA-best swap (optional) ----------
                if did_swap and GSA_best and gsa_candidate is not None:
                    cand = gsa_candidate
                    L = self.layers[cand.layer_id]
                    kb_best = self._key_to_bytes(cand.key)

                    if hash_fn_index >= (cand.resident_level + eff_gap * 2):
                        if cand.seen_level <= cand.resident_level:
                            gsa_candidate = None
                            hash_fn_index += 1
                            continue

                        slot_index_resident = self.hashes.h(cand.layer_id, cand.resident_level, kb_best, L.capacity)
                        probes_this_insert += 1
                        if probes_this_insert >= PROBE_LIMIT:
                            self.layer_reduction(max_layers=active_layers)
                            self.jump_reduction(max_layers=active_layers)
                            return (False, current_key)

                        slot_index_target = self.hashes.h(cand.layer_id, cand.seen_level, key_bytes_current, L.capacity)
                        probes_this_insert += 1
                        if probes_this_insert >= PROBE_LIMIT:
                            self.layer_reduction(max_layers=active_layers)
                            self.jump_reduction(max_layers=active_layers)
                            return (False, current_key)

                        if slot_index_target != slot_index_resident:
                            raise Exception("[GSA-MISALIGNED]")

                        cell = L.at(slot_index_resident)
                        if cell.is_empty or cell.key != cand.key or cell.placement_level != cand.resident_level:
                            raise Exception("[GSA-MISALIGNED]")

                        evicted_key = cell.key
                        evicted_level = cell.placement_level
                        L.unregister_cell_level(evicted_key, evicted_level)
                        cell.key = current_key
                        cell.placement_level = cand.seen_level
                        L.register_cell_level(current_key, cand.seen_level)

                        current_key = evicted_key
                        key_bytes_current = self._key_to_bytes(current_key)
                        gsa_candidate = None
                        break

                # ---------- NEXT hash function index ----------
                hash_fn_index += 1

        self.layer_reduction(max_layers=active_layers)
        self.jump_reduction(max_layers=active_layers)
        return (False, current_key)

    def insert_or_open(self, key: Any, max_layers: int = Active_layers) -> bool:
        """
        Insert key; normally open new layers on failure.
        Here: do NOT exceed max_layers. If we would open layer (max_layers+1), return False.
        """
        ok, pending = self.insert_with_pending(key, active_layers=max_layers)
        if ok:
            return True

        target = pending if pending is not None else key

        while True:
            if len(self.layers) >= max_layers:
                return False  # do NOT create layer max_layers+1

            top_cap = self.layers[-1].capacity if self.layers else self.base_capacity
            new_cap = top_cap * 2
            self.layers.append(Layer(new_cap))

            ok2, pending2 = self.insert_with_pending(target, active_layers=max_layers)
            if ok2:
                return True

            target = pending2 if pending2 is not None else target

    def insert(self, key: Any, active_layers: int = Active_layers) -> bool:
        ok, _ = self.insert_with_pending(key, active_layers=active_layers)
        return ok

    def jump_reduction(self, max_layers, op_limit: int = 10000) -> bool:
        n = len(self.layers)
        if n == 0:
            return False

        if n > max_layers:
            candidate_layers = list(range(n - max_layers, n))
        else:
            candidate_layers = list(range(n))

        ops = 0

        for src_id in candidate_layers:
            if ops >= op_limit:
                break

            src = self.layers[src_id]

            max_level = None
            max_keys: set[Any] | None = None
            for level, keys in src.iter_levels_desc():
                max_level = level
                max_keys = keys
                break

            if not max_keys or max_level is None:
                continue

            if max_level < 4:
                continue

            key = min(list(max_keys))
            kb = self._key_to_bytes(key)

            src_idx = self.hashes.h(src_id, max_level, kb, src.capacity)
            src_cell = src.at(src_idx)
            if src_cell.is_empty or src_cell.key != key or src_cell.placement_level != max_level:
                continue

            for dst_id in reversed(candidate_layers):
                if dst_id == src_id:
                    continue
                dst = self.layers[dst_id]

                for new_level in (1,):
                    if ops >= op_limit:
                        break
                    ops += 1

                    dst_idx = self.hashes.h(dst_id, new_level, kb, dst.capacity)
                    dst_cell = dst.at(dst_idx)

                    if dst_cell.is_empty:
                        src.unregister_cell_level(key, max_level)
                        src_cell.key = None
                        src_cell.placement_level = None
                        src.size -= 1

                        dst_cell.key = key
                        dst_cell.placement_level = new_level
                        dst.size += 1
                        dst.register_cell_level(key, new_level)

                        return True

        return False

    def delete(self, key: Any, d: int = 3) -> bool:
        raise NotImplementedError

    def _global_load_factor(self) -> float:
        total_cap = sum(L.capacity for L in self.layers)
        return (self.num_items / total_cap) if total_cap else 0.0

    def layer_reduction(self, max_layers: int = 3, kick_limit: int = 50) -> bool:
        if len(self.layers) <= max_layers:
            return False

        src_id = 0
        dst_id = len(self.layers) - 1
        src = self.layers[src_id]
        dst = self.layers[dst_id]

        highest = None
        keys_at_level = None
        for level, keys in src.iter_levels_desc():
            highest = level
            keys_at_level = list(keys)
            break

        if not keys_at_level:
            if src.size == 0:
                self.layers.pop(0)
            return False

        key = min(keys_at_level)
        kb = self._key_to_bytes(key)

        src_idx = self.hashes.h(src_id, highest, kb, src.capacity)
        src_cell = src.at(src_idx)
        if src_cell.is_empty or src_cell.key != key or src_cell.placement_level != highest:
            raise Exception(
                f"Stale entry detected! expected key={key!r}@{highest}, "
                f"but got key={src_cell.key!r}@{src_cell.placement_level}, idx={src_idx}, cap={src.capacity}"
            )

        for new_level in range(1, 1 + kick_limit):
            idx = self.hashes.h(dst_id, new_level, kb, dst.capacity)
            dst_cell = dst.at(idx)

            if dst_cell.is_empty:
                src.unregister_cell_level(key, highest)
                src_cell.key = None
                src_cell.placement_level = None
                src.size -= 1

                dst_cell.key = key
                dst_cell.placement_level = new_level
                dst.size += 1
                dst.register_cell_level(key, new_level)

                if src.size == 0:
                    self.layers.pop(0)

                return True

        return False

    def print_loads(self) -> None:
        def pct(x: float) -> str:
            return f"{x * 100:5.1f}%"

        def fmt_int(n: int) -> str:
            return f"{n:,}"

        total_cap = sum(L.capacity for L in self.layers) or 0
        total_items = int(self.num_items)
        global_occ = (total_items / total_cap) if total_cap else 0.0

        total_attempts = getattr(self, "stats_total_attempts", 0)
        total_success = getattr(self, "stats_success_inserts", 0)
        avg_probes_overall = (total_attempts / total_success) if total_success else 0.0

        hl_attempts = getattr(self, "stats_highload_attempts", 0)
        hl_success = getattr(self, "stats_highload_success", 0)
        hl_pct = int(HIGH_LOAD_THRESHOLD * 100)
        avg_probes_high = (hl_attempts / hl_success) if hl_success else 0.0

        print("\n================= FINAL STATE =================")
        print(f"Layers: {len(self.layers)}   Base capacity: {fmt_int(self.base_capacity)}")
        print(f"Global: items={fmt_int(total_items)} / capacity={fmt_int(total_cap)}   occupancy={pct(global_occ)}")
        print("------------------------------------------------")
        print(f"{'L#':>3} | {'capacity':>12} | {'size':>12} | {'occ':>6} | {'max_i':>5}")
        print("-" * 70)

        for i, L in enumerate(self.layers):
            layer_occ = (L.size / L.capacity) if L.capacity else 0.0
            print(
                f"{i:>3} | {fmt_int(L.capacity):>12} | {fmt_int(L.size):>12} | {pct(layer_occ):>6} | "
                f"{L.max_hash_index_used:>5}"
            )

        print("-" * 70)
        print("Status:")
        print(f"  Total inserts:           {fmt_int(total_success)}")
        print(f"  Total probes:            {fmt_int(total_attempts)}")
        print(f"  Avg probes / insert:     {avg_probes_overall:.3f}")
        print("----------------")
        print(f"  High-load threshold:     ≥{hl_pct}%")
        print(f"  High-load inserts:       {fmt_int(hl_success)}")
        print(f"  High-load probes:        {fmt_int(hl_attempts)}")
        print(f"  Avg probes / high-load:  {'N/A' if hl_success == 0 else f'{avg_probes_high:.3f}'}")
        print("----------------")
        print("Rate by GLOBAL load ranges:")
        for (lo, hi) in sorted(self.range_stats.keys()):
            att = self.range_stats[(lo, hi)]["attempts"]
            suc = self.range_stats[(lo, hi)]["success"]
            avg = (att / suc) if suc else None
            print(
                f"  {int(lo*100):02d}–{int(hi*100):02d}% | inserts={fmt_int(suc)} | probes={fmt_int(att)}"
                f" | avg={'N/A' if avg is None else f'{avg:.3f}'}"
            )
        print("================================================\n")

    @staticmethod
    def _key_to_bytes(key: Any) -> bytes:
        if isinstance(key, (bytes, bytearray)):
            return bytes(key)
        return str(key).encode("utf-8", errors="surrogatepass")


# ------------------- helpers -------------------- #

def unique_keys(prefix: str = "k"):
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1


def run_single_simulation(
    base_power: int,
    active_layers: int,
    high_load_threshold: float,
    load_ranges: List[Tuple[float, float]],
):
    global Active_layers, HIGH_LOAD_THRESHOLD, LOAD_RANGES
    Active_layers = active_layers
    HIGH_LOAD_THRESHOLD = high_load_threshold
    LOAD_RANGES = load_ranges

    print("\n================ SIMULATION ================")
    print(f"Active layers target : {Active_layers}")
    print(f"High load threshold  : {HIGH_LOAD_THRESHOLD}")
    print(f"Load ranges measured : {LOAD_RANGES}")
    print("===========================================\n")

    tbl = MultiLayerCuckoo(base_capacity=2 ** base_power)

    gen = unique_keys()
    inserted = 0

    # keep inserting until we would need layer 6 (stop by insert_or_open returning False)
    while True:
        k = next(gen)
        ok = tbl.insert_or_open(k, max_layers=active_layers)
        if not ok:
            break
        inserted += 1

    print(f"Inserted items before stop: {inserted}")
    print(f"Total layers after stop: {len(tbl.layers)}")
    tbl.print_loads()

    for li, L in enumerate(tbl.layers):
        levels = ", ".join(f"{lvl}:{len(keys)}" for lvl, keys in L.iter_levels_desc()) or "-"
        print(f"Layer {li}: size={L.size}/{L.capacity}, max_i_used={L.max_hash_index_used}, levels[{levels}]")

    return {}


# ------------------- run -------------------- #

result = run_single_simulation(
    base_power=18,
    active_layers=5,
    high_load_threshold=0.95,
    load_ranges=[
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.70, 0.80),
        (0.80, 0.85),
        (0.90, 0.94),
        (0.95, 0.98),
        (0.97, 0.98),
        (0.98, 0.99),
    ],
)

print(result)