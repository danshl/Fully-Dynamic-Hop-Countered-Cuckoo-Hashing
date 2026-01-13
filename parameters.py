def total_capacity_5_layers(cap_power: int, layers: int = 5) -> int:
    """
    Returns sum_{i=0..layers-1} 2^(cap_power + i)
    Default layers=5.
    """
    return sum(2 ** (cap_power + i) for i in range(layers))

BASE_POWER = 11
TOTAL_CAPACITY = total_capacity_5_layers(BASE_POWER)