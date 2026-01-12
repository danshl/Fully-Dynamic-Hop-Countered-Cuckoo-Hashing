from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class InsertStats:
    total_probes: int = 0
    total_inserts: int = 0

    highload_probes: int = 0
    highload_inserts: int = 0

    range_stats: Dict[Tuple[float, float], Dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def init_ranges(load_ranges: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Dict[str, int]]:
        return {(lo, hi): {"probes": 0, "inserts": 0} for (lo, hi) in load_ranges}
