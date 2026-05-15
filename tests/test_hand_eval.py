from __future__ import annotations

from hkmahjong_ai.hand_eval import _can_form_melds, can_win, is_seven_pairs
from hkmahjong_ai.tiles import TILE_COUNT, counts_from_tiles


def counts(*tiles: int) -> list[int]:
    return counts_from_tiles(tiles)


def test_can_win_standard_four_melds_and_pair() -> None:
    hand = counts(
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
    )
    assert can_win(hand)


def test_can_form_melds_handles_sequence_branch_without_mutation_leak() -> None:
    meld_tiles = counts(0, 0, 0, 1, 2, 3, 1, 2, 3)
    assert _can_form_melds(tuple(meld_tiles), 3)


def test_seven_pairs_requires_seven_exact_pairs_in_this_project() -> None:
    assert is_seven_pairs(counts(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6))

    # Project rule: a four-of-a-kind is not split into two pairs for seven pairs.
    assert not is_seven_pairs(counts(0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5))


def test_invalid_tile_count_is_not_a_win() -> None:
    impossible_counts = [0] * TILE_COUNT
    impossible_counts[0] = 5
    impossible_counts[1] = 2
    assert not can_win(impossible_counts)
