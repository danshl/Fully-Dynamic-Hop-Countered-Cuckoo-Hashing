from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Set, Iterable, Tuple


@dataclass
class Cell:
    key: Optional[Any] = None
    placement_level: Optional[int] = None

    @property
    def is_empty(self) -> bool:
        return self.key is None


class Layer:
    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self.capacity = capacity
        self.cells: List[Cell] = [Cell() for _ in range(capacity)]
        self.size: int = 0

        # Tracks highest placement level used in this layer.
        self.max_hash_index_used: int = 0

        # Inverted index: level -> set(keys)
        self.level_index: Dict[int, Set[Any]] = {}

    def at(self, index: int) -> Cell:
        return self.cells[index]

    def occupancy(self) -> float:
        return self.size / self.capacity

    def register_level(self, key: Any, level: Optional[int]) -> None:
        if not level or level < 1:
            return
        self.level_index.setdefault(level, set()).add(key)
        if level > self.max_hash_index_used:
            self.max_hash_index_used = level

    def unregister_level(self, key: Any, level: Optional[int]) -> None:
        if not level:
            return
        s = self.level_index.get(level)
        if not s:
            return
        s.discard(key)
        if not s:
            self.level_index.pop(level, None)
            if level == self.max_hash_index_used:
                self.max_hash_index_used = max(self.level_index.keys(), default=0)

    def iter_levels_desc(self) -> Iterable[Tuple[int, Set[Any]]]:
        for lvl in sorted(self.level_index.keys(), reverse=True):
            s = self.level_index[lvl]
            if s:
                yield lvl, s
