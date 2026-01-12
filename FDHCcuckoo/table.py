from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from .config import CuckooConfig
from .hashing import HashFamily
from .layer import Layer
from .stats import InsertStats
from .utils import key_to_bytes


class GSACandidate(NamedTuple):
    layer_id: int
    key: Any
    resident_level: int
    seen_level: int


class MultiLayerCuckoo:
    def __init__(self, base_capacity: int = 128, cfg: Optional[CuckooConfig] = None) -> None:
        assert base_capacity > 0
        self.base_capacity = base_capacity
        self.cfg = cfg or CuckooConfig()

        self.hashes = HashFamily()
        self.layers: List[Layer] = [Layer(base_capacity)]
        self.num_items: int = 0

        self.stats = InsertStats(range_stats=InsertStats.init_ranges(self.cfg.load_ranges))

    # --- load factors ---
    def global_load_factor(self) -> float:
        cap = sum(L.capacity for L in self.layers)
        return (self.num_items / cap) if cap else 0.0

    def active_load_factor(self, k: Optional[int] = None) -> float:
        k = k or self.cfg.active_layers
        layers = self.layers[-k:] if len(self.layers) >= k else self.layers
        cap = sum(L.capacity for L in layers)
        sz = sum(L.size for L in layers)
        return (sz / cap) if cap else 0.0

    def effective_min_gap(self) -> int:
        return 3 if self.active_load_factor() < 0.92 else 8

    # --- query ---
    def contains(self, key: Any) -> bool:
        kb = key_to_bytes(key)
        for layer_id, L in enumerate(self.layers):
            if L.size == 0:
                continue
            max_i = max(L.max_hash_index_used, 1)
            for level in range(1, max_i + 1):
                idx = self.hashes.h(layer_id, level, kb, L.capacity)
                cell = L.at(idx)
                if not cell.is_empty and cell.key == key:
                    return True
        return False

    # --- stats helpers ---
    def _note_ranges_after_insert(self, probes_this_insert: int) -> None:
        occ = self.global_load_factor()
        for (lo, hi), b in self.stats.range_stats.items():
            if lo <= occ <= hi:
                b["probes"] += probes_this_insert
                b["inserts"] += 1

    # --- insert core ---
    def insert_with_pending(self, key: Any) -> Tuple[bool, Optional[Any]]:
        assert key is not None
        cfg = self.cfg

        current_key = key
        current_kb = key_to_bytes(current_key)

        probes = 0
        gsa_candidate: Optional[GSACandidate] = None

        while probes < cfg.probe_limit:
            level = 1

            while True:
                if self.global_load_factor() >= cfg.occ_stop:
                    return False, current_key

                eff_gap = self.effective_min_gap()

                # choose active layers (prefer not-too-full)
                if len(self.layers) > cfg.active_layers:
                    offset = len(self.layers) - cfg.active_layers
                    pairs = [(offset + i, L) for i, L in enumerate(self.layers[-cfg.active_layers:])]
                else:
                    pairs = list(enumerate(self.layers))

                not_full = [(lid, L) for (lid, L) in pairs if L.occupancy() < 0.998]
                if not_full:
                    pairs = not_full

                # per-level index cache
                idx_cache: Dict[Tuple[int, int], int] = {}

                def get_index(layer_id: int, L: Layer) -> Optional[int]:
                    nonlocal probes
                    ck = (layer_id, level)
                    if ck in idx_cache:
                        return idx_cache[ck]
                    idx = self.hashes.h(layer_id, level, current_kb, L.capacity)
                    probes += 1
                    if probes >= cfg.probe_limit:
                        self.layer_reduction(max_layers=cfg.active_layers)
                        self.jump_reduction(max_layers=cfg.active_layers)
                        return None
                    idx_cache[ck] = idx
                    return idx

                # A) empty slot (top-first)
                for layer_id, L in reversed(pairs):
                    idx = get_index(layer_id, L)
                    if idx is None:
                        return False, current_key
                    cell = L.at(idx)
                    if cell.is_empty:
                        cell.key = current_key
                        cell.placement_level = level
                        L.size += 1
                        self.num_items += 1
                        L.register_level(current_key, level)

                        self.stats.total_probes += probes
                        self.stats.total_inserts += 1
                        self._note_ranges_after_insert(probes)

                        if self.active_load_factor(cfg.active_layers) >= cfg.high_load_threshold:
                            self.stats.highload_probes += probes
                            self.stats.highload_inserts += 1

                        if self.stats.total_inserts % cfg.maintenance_period == 0:
                            self.layer_reduction(max_layers=cfg.active_layers)
                            self.jump_reduction(max_layers=cfg.active_layers)

                        return True, None

                # B) swap if allowed (scan active)
                did_swap = False
                gsa_candidate = None

                for layer_id, L in pairs:
                    idx = get_index(layer_id, L)
                    if idx is None:
                        return False, current_key

                    cell = L.at(idx)
                    if cell.is_empty:
                        continue

                    resident_level = cell.placement_level or 1
                    if level >= resident_level + eff_gap:
                        evicted_key = cell.key
                        evicted_level = cell.placement_level

                        L.unregister_level(evicted_key, evicted_level)
                        cell.key = current_key
                        cell.placement_level = level
                        L.register_level(current_key, level)

                        current_key = evicted_key
                        current_kb = key_to_bytes(current_key)
                        did_swap = True
                        break

                    # track best candidate (smallest resident_level)
                    if cfg.gsa_best:
                        if (gsa_candidate is None) or (resident_level < gsa_candidate.resident_level):
                            gsa_candidate = GSACandidate(layer_id, cell.key, resident_level, level)

                if did_swap:
                    break

                # C) optional GSA-best (only when no swap happened)
                if cfg.gsa_best and gsa_candidate is not None:
                    cand = gsa_candidate
                    L = self.layers[cand.layer_id]

                    if level >= cand.resident_level + eff_gap * 2 and cand.seen_level > cand.resident_level:
                        kb_best = key_to_bytes(cand.key)

                        idx_resident = self.hashes.h(cand.layer_id, cand.resident_level, kb_best, L.capacity)
                        probes += 1
                        if probes >= cfg.probe_limit:
                            self.layer_reduction(max_layers=cfg.active_layers)
                            self.jump_reduction(max_layers=cfg.active_layers)
                            return False, current_key

                        idx_target = self.hashes.h(cand.layer_id, cand.seen_level, current_kb, L.capacity)
                        probes += 1
                        if probes >= cfg.probe_limit:
                            self.layer_reduction(max_layers=cfg.active_layers)
                            self.jump_reduction(max_layers=cfg.active_layers)
                            return False, current_key

                        if idx_target == idx_resident:
                            cell = L.at(idx_resident)
                            if (not cell.is_empty) and cell.key == cand.key and cell.placement_level == cand.resident_level:
                                evicted_key = cell.key
                                evicted_level = cell.placement_level

                                L.unregister_level(evicted_key, evicted_level)
                                cell.key = current_key
                                cell.placement_level = cand.seen_level
                                L.register_level(current_key, cand.seen_level)

                                current_key = evicted_key
                                current_kb = key_to_bytes(current_key)
                                break

                level += 1

        self.layer_reduction(max_layers=cfg.active_layers)
        self.jump_reduction(max_layers=cfg.active_layers)
        return False, current_key

    def insert_or_open(self, key: Any) -> bool:
        ok, pending = self.insert_with_pending(key)
        if ok:
            return True

        target = pending if pending is not None else key
        while True:
            if len(self.layers) >= self.cfg.active_layers:
                return False

            top_cap = self.layers[-1].capacity
            self.layers.append(Layer(top_cap * 2))

            ok2, pending2 = self.insert_with_pending(target)
            if ok2:
                return True
            target = pending2 if pending2 is not None else target

    # --- maintenance (kept identical behavior, just isolated) ---
    def jump_reduction(self, max_layers: int, op_limit: int = 10000) -> bool:
        n = len(self.layers)
        if n == 0:
            return False

        candidate = list(range(n - max_layers, n)) if n > max_layers else list(range(n))
        ops = 0

        for src_id in candidate:
            if ops >= op_limit:
                break
            src = self.layers[src_id]

            max_level = None
            max_keys = None
            for lvl, keys in src.iter_levels_desc():
                max_level, max_keys = lvl, keys
                break

            if not max_keys or max_level is None or max_level < 4:
                continue

            key = min(list(max_keys))
            kb = key_to_bytes(key)

            src_idx = self.hashes.h(src_id, max_level, kb, src.capacity)
            src_cell = src.at(src_idx)
            if src_cell.is_empty or src_cell.key != key or src_cell.placement_level != max_level:
                continue

            for dst_id in reversed(candidate):
                if dst_id == src_id:
                    continue
                dst = self.layers[dst_id]

                ops += 1
                dst_idx = self.hashes.h(dst_id, 1, kb, dst.capacity)
                dst_cell = dst.at(dst_idx)

                if dst_cell.is_empty:
                    src.unregister_level(key, max_level)
                    src_cell.key = None
                    src_cell.placement_level = None
                    src.size -= 1

                    dst_cell.key = key
                    dst_cell.placement_level = 1
                    dst.size += 1
                    dst.register_level(key, 1)
                    return True

        return False

    def layer_reduction(self, max_layers: int = 3, kick_limit: int = 50) -> bool:
        if len(self.layers) <= max_layers:
            return False

        src = self.layers[0]
        dst = self.layers[-1]

        highest = None
        keys_at_level = None
        for lvl, keys in src.iter_levels_desc():
            highest = lvl
            keys_at_level = list(keys)
            break

        if not keys_at_level:
            if src.size == 0:
                self.layers.pop(0)
            return False

        key = min(keys_at_level)
        kb = key_to_bytes(key)

        src_idx = self.hashes.h(0, highest, kb, src.capacity)
        src_cell = src.at(src_idx)
        if src_cell.is_empty or src_cell.key != key or src_cell.placement_level != highest:
            return False

        for new_level in range(1, 1 + kick_limit):
            idx = self.hashes.h(len(self.layers) - 1, new_level, kb, dst.capacity)
            dst_cell = dst.at(idx)
            if dst_cell.is_empty:
                src.unregister_level(key, highest)
                src_cell.key = None
                src_cell.placement_level = None
                src.size -= 1

                dst_cell.key = key
                dst_cell.placement_level = new_level
                dst.size += 1
                dst.register_level(key, new_level)

                if src.size == 0:
                    self.layers.pop(0)
                return True

        return False

    # --- reporting ---
    def print_loads(self) -> None:
        def pct(x: float) -> str:
            return f"{x * 100:5.1f}%"

        def fmt_int(n: int) -> str:
            return f"{n:,}"

        total_cap = sum(L.capacity for L in self.layers) or 0
        total_items = int(self.num_items)
        occ = (total_items / total_cap) if total_cap else 0.0

        avg_all = (self.stats.total_probes / self.stats.total_inserts) if self.stats.total_inserts else 0.0
        avg_hl = (self.stats.highload_probes / self.stats.highload_inserts) if self.stats.highload_inserts else 0.0

        print("\n================= FINAL STATE =================")
        print(f"Layers: {len(self.layers)}   Base capacity: {fmt_int(self.base_capacity)}")
        print(f"Global: items={fmt_int(total_items)} / capacity={fmt_int(total_cap)}   occupancy={pct(occ)}")
        print("------------------------------------------------")
        print(f"{'L#':>3} | {'capacity':>12} | {'size':>12} | {'occ':>6} | {'max_i':>5}")
        print("-" * 70)
        for i, L in enumerate(self.layers):
            print(f"{i:>3} | {fmt_int(L.capacity):>12} | {fmt_int(L.size):>12} | {pct(L.occupancy()):>6} | {L.max_hash_index_used:>5}")
        print("-" * 70)

        print("Status:")
        print(f"  Total inserts:          {fmt_int(self.stats.total_inserts)}")
        print(f"  Total probes:           {fmt_int(self.stats.total_probes)}")
        print(f"  Avg probes / insert:    {avg_all:.3f}")
        print("----------------")
        print(f"  High-load threshold:    ≥{int(self.cfg.high_load_threshold*100)}%")
        print(f"  High-load inserts:      {fmt_int(self.stats.highload_inserts)}")
        print(f"  High-load probes:       {fmt_int(self.stats.highload_probes)}")
        print(f"  Avg probes / high-load: {'N/A' if self.stats.highload_inserts == 0 else f'{avg_hl:.3f}'}")
        print("----------------")

        print("Rate by GLOBAL load ranges:")
        for (lo, hi) in sorted(self.stats.range_stats.keys()):
            p = self.stats.range_stats[(lo, hi)]["probes"]
            s = self.stats.range_stats[(lo, hi)]["inserts"]
            avg = (p / s) if s else None
            print(
                f"  {int(lo*100):02d}–{int(hi*100):02d}% | inserts={fmt_int(s)} | probes={fmt_int(p)}"
                f" | avg={'N/A' if avg is None else f'{avg:.3f}'}"
            )
        print("================================================\n")
