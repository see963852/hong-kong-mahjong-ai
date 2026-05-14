from __future__ import annotations

from collections import Counter
from typing import Iterable

SUITS = ("萬", "筒", "索")
WINDS = ("東", "南", "西", "北")
DRAGONS = ("中", "發", "白")
TILE_COUNT = 34


def tile_name(tile: int) -> str:
    if 0 <= tile <= 8:
        return f"{tile + 1}萬"
    if 9 <= tile <= 17:
        return f"{tile - 8}筒"
    if 18 <= tile <= 26:
        return f"{tile - 17}索"
    if 27 <= tile <= 30:
        return WINDS[tile - 27]
    if 31 <= tile <= 33:
        return DRAGONS[tile - 31]
    raise ValueError(f"invalid tile: {tile}")


def tile_sort_key(tile: int) -> tuple[int, int]:
    return (tile // 9 if tile < 27 else 3, tile)


def sorted_tiles(tiles: Iterable[int]) -> list[int]:
    return sorted(tiles, key=tile_sort_key)


def all_wall() -> list[int]:
    return [tile for tile in range(TILE_COUNT) for _ in range(4)]


def counts_from_tiles(tiles: Iterable[int]) -> list[int]:
    counts = [0] * TILE_COUNT
    for tile in tiles:
        counts[tile] += 1
    return counts


def names_from_tiles(tiles: Iterable[int]) -> str:
    return " ".join(tile_name(tile) for tile in sorted_tiles(tiles))


def compact_counts(tiles: Iterable[int]) -> str:
    counter = Counter(tiles)
    return " ".join(f"{tile_name(tile)}x{counter[tile]}" for tile in sorted(counter))


def is_suited(tile: int) -> bool:
    return 0 <= tile < 27


def suit_index(tile: int) -> int:
    if not is_suited(tile):
        return -1
    return tile // 9


def rank(tile: int) -> int:
    if not is_suited(tile):
        return -1
    return tile % 9 + 1
