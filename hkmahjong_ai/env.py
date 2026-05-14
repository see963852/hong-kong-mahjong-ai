from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Protocol

from .hand_eval import can_win, chow_options, winning_with_tile
from .tiles import all_wall, counts_from_tiles, names_from_tiles, sorted_tiles, tile_name


class ClaimAgent(Protocol):
    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        ...


@dataclass
class Meld:
    kind: str
    tiles: list[int]
    from_player: int | None = None

    def label(self) -> str:
        return f"{self.kind}:{names_from_tiles(self.tiles)}"


@dataclass
class PlayerState:
    hand: list[int] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)

    def counts(self) -> list[int]:
        return counts_from_tiles(self.hand)


@dataclass
class GameResult:
    winner: int | None
    loser: int | None
    kind: str
    turns: int
    wall_remaining: int
    message: str


class HKMahjongEnv:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.players = [PlayerState() for _ in range(4)]
        self.wall: list[int] = []
        self.current_player = 0
        self.dealer = 0
        self.turns = 0
        self.done = False
        self.result: GameResult | None = None
        self.last_discard: tuple[int, int] | None = None
        self.log: list[str] = []

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        self.players = [PlayerState() for _ in range(4)]
        self.wall = all_wall()
        self.rng.shuffle(self.wall)
        self.current_player = self.dealer
        self.turns = 0
        self.done = False
        self.result = None
        self.last_discard = None
        self.log = []

        for _ in range(13):
            for player in self.players:
                player.hand.append(self.wall.pop())
        self.players[self.dealer].hand.append(self.wall.pop())
        for player in self.players:
            player.hand = sorted_tiles(player.hand)
        self._log(f"開局，莊家 P{self.dealer + 1}")
        return self.observation()

    def observation(self, player_id: int | None = None) -> dict[str, Any]:
        pid = self.current_player if player_id is None else player_id
        player = self.players[pid]
        return {
            "player": pid,
            "current_player": self.current_player,
            "hand": list(player.hand),
            "hand_counts": player.counts(),
            "open_melds": len(player.melds),
            "melds": [meld.tiles for meld in player.melds],
            "discards": [list(p.discards) for p in self.players],
            "wall_remaining": len(self.wall),
            "turns": self.turns,
            "done": self.done,
        }

    def legal_discards(self, player_id: int | None = None) -> list[int]:
        pid = self.current_player if player_id is None else player_id
        return sorted(set(self.players[pid].hand))

    def can_self_win(self, player_id: int | None = None) -> bool:
        pid = self.current_player if player_id is None else player_id
        player = self.players[pid]
        return can_win(player.counts(), len(player.melds))

    def declare_self_win(self, player_id: int | None = None) -> GameResult:
        pid = self.current_player if player_id is None else player_id
        return self._finish(pid, None, "self_draw", f"P{pid + 1} 自摸")

    def discard(self, tile: int, agents: list[ClaimAgent] | None = None) -> GameResult | None:
        if self.done:
            return self.result
        player = self.players[self.current_player]
        if tile not in player.hand:
            raise ValueError(f"P{self.current_player + 1} does not hold {tile_name(tile)}")

        player.hand.remove(tile)
        player.discards.append(tile)
        discarder = self.current_player
        self.last_discard = (discarder, tile)
        self.turns += 1
        self._log(f"P{discarder + 1} 打出 {tile_name(tile)}")

        result = self._resolve_claims(discarder, tile, agents)
        if result is not None:
            return result

        return self._draw_next((discarder + 1) % 4)

    def play_ai_turn(self, agents: list[Any]) -> GameResult | None:
        if self.done:
            return self.result
        player_id = self.current_player
        if self.can_self_win(player_id):
            return self.declare_self_win(player_id)
        legal = self.legal_discards(player_id)
        tile = agents[player_id].choose_discard(self.observation(player_id), legal)
        return self.discard(tile, agents)

    def snapshot(self) -> dict[str, Any]:
        return {
            "players": [
                {
                    "hand": list(player.hand),
                    "melds": [meld.label() for meld in player.melds],
                    "discards": list(player.discards),
                }
                for player in self.players
            ],
            "current_player": self.current_player,
            "wall_remaining": len(self.wall),
            "turns": self.turns,
            "done": self.done,
            "result": self.result,
            "log": list(self.log[-30:]),
        }

    def _resolve_claims(
        self, discarder: int, tile: int, agents: list[ClaimAgent] | None
    ) -> GameResult | None:
        win_candidates = []
        for offset in range(1, 4):
            pid = (discarder + offset) % 4
            player = self.players[pid]
            if winning_with_tile(player.counts(), tile, len(player.melds)):
                win_candidates.append(pid)
        if win_candidates:
            winner = win_candidates[0]
            return self._finish(winner, discarder, "discard_win", f"P{winner + 1} 食糊 {tile_name(tile)}")

        if agents is None:
            return None

        claim_options_by_player: list[tuple[int, list[dict[str, Any]]]] = []
        for offset in range(1, 4):
            pid = (discarder + offset) % 4
            player = self.players[pid]
            counts = player.counts()
            options: list[dict[str, Any]] = []
            if counts[tile] >= 2:
                options.append({"kind": "pong", "tiles": [tile, tile, tile], "consume": [tile, tile]})
            if offset == 1:
                for seq in chow_options(counts, tile):
                    consume = list(seq)
                    consume.remove(tile)
                    options.append({"kind": "chow", "tiles": list(seq), "consume": consume})
            if options:
                claim_options_by_player.append((pid, options))

        for preferred_kind in ("pong", "chow"):
            for pid, options in claim_options_by_player:
                filtered = [opt for opt in options if opt["kind"] == preferred_kind]
                if not filtered:
                    continue
                state = self.observation(pid) | {"discarder": discarder, "discard_tile": tile}
                chosen = agents[pid].choose_claim(state, filtered)
                if chosen is not None:
                    self._apply_claim(pid, discarder, chosen)
                    return None
        return None

    def _apply_claim(self, player_id: int, discarder: int, claim: dict[str, Any]) -> None:
        player = self.players[player_id]
        for tile in claim["consume"]:
            player.hand.remove(tile)
        player.melds.append(Meld(claim["kind"], sorted_tiles(claim["tiles"]), discarder))
        self.current_player = player_id
        self._log(f"P{player_id + 1} {claim['kind']} {names_from_tiles(claim['tiles'])}")

    def _draw_next(self, player_id: int) -> GameResult | None:
        self.current_player = player_id
        if not self.wall:
            return self._finish(None, None, "draw", "牌牆摸完，流局")
        tile = self.wall.pop()
        player = self.players[player_id]
        player.hand.append(tile)
        player.hand = sorted_tiles(player.hand)
        self._log(f"P{player_id + 1} 摸牌")
        return None

    def _finish(self, winner: int | None, loser: int | None, kind: str, message: str) -> GameResult:
        self.done = True
        self.result = GameResult(winner, loser, kind, self.turns, len(self.wall), message)
        self._log(message)
        return self.result

    def _log(self, message: str) -> None:
        self.log.append(message)
