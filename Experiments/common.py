from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

BINS: List[Tuple[float, float]] = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.85),
    (0.90, 0.94),
    (0.95, 0.97),
    (0.97, 0.98),
    (0.98, 0.99),
]

@dataclass
class ModelResult:
    name: str
    avg_probes: List[Optional[float]]   # <-- זה השדה שהגרפים משתמשים בו
    fail_load: float
