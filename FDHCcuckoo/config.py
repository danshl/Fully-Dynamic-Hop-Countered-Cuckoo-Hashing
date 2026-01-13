from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class CuckooConfig:
    active_layers: int = 5
    gsa_best: bool = True
    maintenance_period: int = 100
    probe_limit: int = 1000
    occ_stop: float = 1.0
    high_load_threshold: float = 0.95

    load_ranges: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.load_ranges is None:
            object.__setattr__(
                self,
                "load_ranges",
                [
                    (0.90, 0.92),
                    (0.92, 0.94),
                    (0.94, 0.95),
                    (0.95, 0.96),
                    (0.96, 0.97),
                    (0.97, 0.98),
                    (0.98, 0.99),
                ],
            )
