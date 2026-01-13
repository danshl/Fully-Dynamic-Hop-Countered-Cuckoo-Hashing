from __future__ import annotations
import random
from typing import List, Tuple, Optional

from FDHCcuckoo.config import CuckooConfig
from FDHCcuckoo.table import MultiLayerCuckoo
from FDHCcuckoo.utils import unique_keys


def run_single_simulation(
    base_power: int = 13,
    active_layers: int = 5,
    high_load_threshold: float = 0.95,
    load_ranges: Optional[List[Tuple[float, float]]] = None,
    seed: int = 0,
    active_layer_threshold: float = 0.998,  # NEW
):
    random.seed(seed)

    cfg = CuckooConfig(
        active_layers=active_layers,
        high_load_threshold=high_load_threshold,
        load_ranges=load_ranges,
    )

    tbl = MultiLayerCuckoo(base_capacity=2 ** base_power, cfg=cfg)

    gen = unique_keys()
    inserted = 0

    while True:
        k = next(gen)
        if not tbl.insert_or_open(k, active_layer_threshold=active_layer_threshold):  # NEW
            break
        inserted += 1

    print(f"Inserted items before stop: {inserted}")
    print(f"Total layers after stop: {len(tbl.layers)}")
    tbl.print_loads()

    return tbl


if __name__ == "__main__":
    run_single_simulation()
