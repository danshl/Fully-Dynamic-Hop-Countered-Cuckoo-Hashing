from __future__ import annotations
from typing import Any, Iterator


def key_to_bytes(key: Any) -> bytes:
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    return str(key).encode("utf-8", errors="surrogatepass")


def unique_keys(prefix: str = "k") -> Iterator[str]:
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1
