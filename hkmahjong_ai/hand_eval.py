from __future__ import annotations

from functools import lru_cache

from .tiles import TILE_COUNT, is_suited


def can_win(counts: list[int], open_melds: int = 0) -> bool:
    """Standard 4 melds + 1 pair, plus closed seven pairs."""
    concealed_tiles = sum(counts)
    if open_melds == 0 and concealed_tiles == 14 and is_seven_pairs(counts):
        return True

    needed_melds = 4 - open_melds
    if needed_melds < 0:
        return False
    if concealed_tiles != needed_melds * 3 + 2:
        return False

    counts_tuple = tuple(counts)
    for pair in range(TILE_COUNT):
        if counts[pair] >= 2:
            remaining = list(counts_tuple)
            remaining[pair] -= 2
            if _can_form_melds(tuple(remaining), needed_melds):
                return True
    return False


def is_seven_pairs(counts: list[int]) -> bool:
    return sum(counts) == 14 and sum(1 for count in counts if count == 2) == 7


def classify_win(counts: list[int], open_melds: int = 0) -> str:
    if open_melds == 0 and is_seven_pairs(counts):
        return "七對子"
    if can_win(counts, open_melds):
        return "四面一對"
    return ""


@lru_cache(maxsize=200_000)
def _can_form_melds(counts: tuple[int, ...], melds_left: int) -> bool:
    if melds_left == 0:
        return sum(counts) == 0

    try:
        first = next(i for i, c in enumerate(counts) if c)
    except StopIteration:
        return melds_left == 0

    data = list(counts)
    if data[first] >= 3:
        data[first] -= 3
        if _can_form_melds(tuple(data), melds_left - 1):
            return True
        data[first] += 3

    if is_suited(first) and first % 9 <= 6 and data[first + 1] > 0 and data[first + 2] > 0:
        data[first] -= 1
        data[first + 1] -= 1
        data[first + 2] -= 1
        if _can_form_melds(tuple(data), melds_left - 1):
            return True

    return False


def winning_with_tile(hand_counts: list[int], tile: int, open_melds: int = 0) -> bool:
    counts = list(hand_counts)
    counts[tile] += 1
    return can_win(counts, open_melds)


def chow_options(hand_counts: list[int], tile: int) -> list[tuple[int, int, int]]:
    if not is_suited(tile):
        return []

    options: list[tuple[int, int, int]] = []
    base = (tile // 9) * 9
    rank = tile % 9
    for start_rank in (rank - 2, rank - 1, rank):
        if not (0 <= start_rank <= 6):
            continue
        seq = (base + start_rank, base + start_rank + 1, base + start_rank + 2)
        needed = [t for t in seq if t != tile]
        if all(hand_counts[t] > 0 for t in needed):
            options.append(seq)
    return options


def hand_potential(counts: list[int], open_melds: int = 0, allow_chow: bool = False) -> int:
    """Small heuristic score for incomplete hands. Higher is better."""
    score = open_melds * 8
    data = list(counts)

    for tile, count in enumerate(data):
        if count >= 3:
            score += 8
        elif count == 2:
            score += 4

    if allow_chow:
        for suit in range(3):
            offset = suit * 9
            for i in range(7):
                score += min(data[offset + i], data[offset + i + 1], data[offset + i + 2]) * 7
            for i in range(8):
                if data[offset + i] and data[offset + i + 1]:
                    score += 2
            for i in range(7):
                if data[offset + i] and data[offset + i + 2]:
                    score += 1

    for tile, count in enumerate(data):
        if count == 1:
            if tile >= 27:
                score -= 2
            else:
                left = tile % 9 > 0 and data[tile - 1] > 0
                right = tile % 9 < 8 and data[tile + 1] > 0
                if not left and not right:
                    score -= 1
    return score
