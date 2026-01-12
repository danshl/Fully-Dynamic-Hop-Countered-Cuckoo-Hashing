from __future__ import annotations
import hashlib


class HashFamily:
    @staticmethod
    def h(layer_id: int, level: int, key_bytes: bytes, capacity: int) -> int:
        # Double hashing with per-layer salt.
        assert level >= 1

        m1 = hashlib.sha256()
        m1.update(b"\xA5")
        m1.update(layer_id.to_bytes(4, "little", signed=False))
        m1.update(key_bytes)
        h1 = int.from_bytes(m1.digest()[:8], "little", signed=False)

        m2 = hashlib.sha256()
        m2.update(b"\x5A")
        m2.update(layer_id.to_bytes(4, "little", signed=False))
        m2.update(key_bytes)
        h2 = int.from_bytes(m2.digest()[:8], "little", signed=False) | 1

        return (h1 + level * h2) % capacity
