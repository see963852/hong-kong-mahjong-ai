from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import TYPE_CHECKING, Any, Protocol

from .hand_eval import can_win, classify_win, winning_with_tile
from .tiles import TILE_COUNT, all_wall, counts_from_tiles, names_from_tiles, sorted_tiles, tile_name

if TYPE_CHECKING:
    import torch


class ClaimAgent(Protocol):
    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        ...

    def choose_kong(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        ...


@dataclass(frozen=True)
class RuleConfig:
    allow_multi_win: bool = True
    allow_rob_added_kong: bool = True
    allow_rob_concealed_kong: bool = False
    ron_points: int = 2
    self_draw_points_each: int = 2
    melded_kong_points: int = 3
    added_kong_points_each: int = 1
    concealed_kong_points_each: int = 2


@dataclass
class Meld:
    kind: str
    tiles: list[int]
    from_player: int | None = None
    kong_type: str | None = None

    def label(self) -> str:
        kind = f"{self.kong_type}_{self.kind}" if self.kong_type else self.kind
        return f"{kind}:{names_from_tiles(self.tiles)}"


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
    pattern: str = ""
    winners: list[int] = field(default_factory=list)
    losers: list[int] = field(default_factory=list)
    score_delta: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    next_dealer: int | None = None

    def __post_init__(self) -> None:
        if not self.winners and self.winner is not None:
            self.winners = [self.winner]
        if not self.losers and self.loser is not None:
            self.losers = [self.loser]


@dataclass
class ScoreEvent:
    kind: str
    player: int
    payer: int | None
    amount: int
    tile: int | None = None


class HKMahjongEnv:
    def __init__(self, seed: int | None = None, rules: RuleConfig | None = None):
        self.rng = random.Random(seed)
        self.rules = rules or RuleConfig()
        self.players = [PlayerState() for _ in range(4)]
        self.wall: list[int] = []
        self.current_player = 0
        self.dealer = 0
        self.turns = 0
        self.done = False
        self.result: GameResult | None = None
        self.last_discard: tuple[int, int] | None = None
        self.hand_score_delta = [0, 0, 0, 0]
        self.scores = [0, 0, 0, 0]
        self.score_events: list[ScoreEvent] = []
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
        self.hand_score_delta = [0, 0, 0, 0]
        self.score_events = []
        self.log = []

        for _ in range(13):
            for player in self.players:
                player.hand.append(self.wall.pop())
        self.players[self.dealer].hand.append(self.wall.pop())
        for player in self.players:
            player.hand = sorted_tiles(player.hand)
        self._log(f"開局，莊家 P{self.dealer + 1}，牌組 136 張")
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
            "meld_counts": [len(p.melds) for p in self.players],
            "all_meld_tile_counts": [
                counts_from_tiles(tile for meld in p.melds for tile in meld.tiles)
                for p in self.players
            ],
            "discards": [list(p.discards) for p in self.players],
            "wall_remaining": len(self.wall),
            "turns": self.turns,
            "done": self.done,
            "scores": list(self.scores),
            "hand_score_delta": list(self.hand_score_delta),
        }

    def legal_discards(self, player_id: int | None = None) -> list[int]:
        pid = self.current_player if player_id is None else player_id
        return sorted(set(self.players[pid].hand))

    def legal_action_mask(self, player_id: int) -> "torch.Tensor":
        from .rl_encoder import claim_action_mask, discard_action_mask

        if self.done:
            return discard_action_mask([])
        if player_id == self.current_player:
            return discard_action_mask(self.legal_discards(player_id))
        if self.last_discard is not None:
            discarder, tile = self.last_discard
            discard_is_pending = bool(self.players[discarder].discards and self.players[discarder].discards[-1] == tile)
            if discard_is_pending and player_id != discarder:
                options = self._claim_options_for_player(player_id, discarder, tile)
                if options:
                    return claim_action_mask(options)
        return discard_action_mask(self.legal_discards(player_id))

    def can_self_win(self, player_id: int | None = None) -> bool:
        pid = self.current_player if player_id is None else player_id
        player = self.players[pid]
        return can_win(player.counts(), len(player.melds))

    def declare_self_win(self, player_id: int | None = None) -> GameResult:
        if self.done and self.result is not None:
            return self.result
        pid = self.current_player if player_id is None else player_id
        player = self.players[pid]
        if not self.can_self_win(pid):
            raise ValueError(f"P{pid + 1} cannot self-win with current hand")
        pattern = classify_win(player.counts(), len(player.melds))
        self._pay_all_others(pid, self.rules.self_draw_points_each, "self_draw", None)
        return self._finish(pid, None, "self_draw", f"P{pid + 1} 自摸", pattern, winners=[pid], next_dealer=pid)

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

        result, claimed = self._resolve_claims(discarder, tile, agents)
        if result is not None:
            return result
        if claimed:
            return None
        self.last_discard = None
        return self._draw_next((discarder + 1) % 4)

    def play_ai_turn(self, agents: list[Any]) -> GameResult | None:
        if self.done:
            return self.result
        player_id = self.current_player
        if self.can_self_win(player_id):
            return self.declare_self_win(player_id)
        kong_options = self.kong_options(player_id)
        if kong_options and hasattr(agents[player_id], "choose_kong"):
            chosen_kong = agents[player_id].choose_kong(self.observation(player_id), kong_options)
            if chosen_kong is not None:
                return self.declare_kong(player_id, chosen_kong, agents)
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
            "scores": list(self.scores),
            "hand_score_delta": list(self.hand_score_delta),
            "score_events": list(self.score_events),
            "log": list(self.log[-30:]),
        }

    def concealed_kong_options(self, player_id: int | None = None) -> list[dict[str, Any]]:
        pid = self.current_player if player_id is None else player_id
        counts = self.players[pid].counts()
        return [
            {
                "kind": "kong",
                "kong_type": "concealed",
                "tiles": [tile, tile, tile, tile],
                "consume": [tile, tile, tile, tile],
                "tile": tile,
            }
            for tile, count in enumerate(counts)
            if count == 4
        ]

    def added_kong_options(self, player_id: int | None = None) -> list[dict[str, Any]]:
        pid = self.current_player if player_id is None else player_id
        player = self.players[pid]
        counts = player.counts()
        options: list[dict[str, Any]] = []
        for meld_index, meld in enumerate(player.melds):
            if meld.kind != "pong":
                continue
            tile = meld.tiles[0]
            if counts[tile] > 0:
                options.append(
                    {
                        "kind": "kong",
                        "kong_type": "added",
                        "tiles": [tile, tile, tile, tile],
                        "consume": [tile],
                        "tile": tile,
                        "meld_index": meld_index,
                    }
                )
        return options

    def kong_options(self, player_id: int | None = None) -> list[dict[str, Any]]:
        return self.added_kong_options(player_id) + self.concealed_kong_options(player_id)

    def declare_kong(
        self,
        player_id: int,
        option: dict[str, Any],
        agents: list[ClaimAgent] | None = None,
    ) -> GameResult | None:
        kong_type = option.get("kong_type")
        if kong_type == "concealed":
            return self.declare_concealed_kong(player_id, int(option["tile"]), agents)
        if kong_type == "added":
            return self.declare_added_kong(player_id, int(option["tile"]), int(option["meld_index"]), agents)
        raise ValueError(f"unknown kong option: {option}")

    def _resolve_claims(
        self, discarder: int, tile: int, agents: list[ClaimAgent] | None
    ) -> tuple[GameResult | None, bool]:
        win_candidates: list[int] = []
        for offset in range(1, 4):
            pid = (discarder + offset) % 4
            player = self.players[pid]
            if winning_with_tile(player.counts(), tile, len(player.melds)):
                win_candidates.append(pid)
        if win_candidates:
            winners = win_candidates if self.rules.allow_multi_win else [win_candidates[0]]
            return (self._finish_discard_win(winners, discarder, tile), False)

        if agents is None:
            return None, False

        claim_options_by_player: list[tuple[int, list[dict[str, Any]]]] = []
        for offset in range(1, 4):
            pid = (discarder + offset) % 4
            options = self._claim_options_for_player(pid, discarder, tile)
            if options:
                claim_options_by_player.append((pid, options))

        for preferred_kind in ("kong", "pong"):
            for pid, options in claim_options_by_player:
                filtered = [opt for opt in options if opt["kind"] == preferred_kind]
                if not filtered:
                    continue
                state = self.observation(pid) | {"discarder": discarder, "discard_tile": tile}
                chosen = agents[pid].choose_claim(state, filtered)
                if chosen is not None:
                    result = self._apply_claim(pid, discarder, chosen)
                    return result, True
        return None, False

    def _claim_options_for_player(self, player_id: int, discarder: int, tile: int) -> list[dict[str, Any]]:
        if player_id == discarder or not (0 <= tile < TILE_COUNT):
            return []
        counts = self.players[player_id].counts()
        options: list[dict[str, Any]] = []
        if counts[tile] >= 3:
            options.append(
                {
                    "kind": "kong",
                    "tiles": [tile, tile, tile, tile],
                    "consume": [tile, tile, tile],
                    "discard_tile": tile,
                }
            )
        if counts[tile] >= 2:
            options.append(
                {
                    "kind": "pong",
                    "tiles": [tile, tile, tile],
                    "consume": [tile, tile],
                    "discard_tile": tile,
                }
            )
        return options

    def _apply_claim(self, player_id: int, discarder: int, claim: dict[str, Any]) -> GameResult | None:
        player = self.players[player_id]
        for tile in claim["consume"]:
            player.hand.remove(tile)
        kong_type = "melded" if claim["kind"] == "kong" else None
        player.melds.append(Meld(claim["kind"], sorted_tiles(claim["tiles"]), discarder, kong_type))

        discard_tile = self.last_discard[1] if self.last_discard is not None else claim.get("discard_tile")
        if self.players[discarder].discards and self.players[discarder].discards[-1] == discard_tile:
            self.players[discarder].discards.pop()
        self.last_discard = None

        self.current_player = player_id
        action = {"chow": "食", "pong": "碰", "kong": "槓"}.get(claim["kind"], claim["kind"])
        self._log(f"P{player_id + 1} {action} {names_from_tiles(claim['tiles'])}")
        if claim["kind"] == "kong":
            self._pay(discarder, player_id, self.rules.melded_kong_points, "melded_kong", discard_tile)
            return self._draw_replacement(player_id)
        return None

    def declare_concealed_kong(
        self,
        player_id: int,
        tile: int,
        agents: list[ClaimAgent] | None = None,
    ) -> GameResult | None:
        if self.done:
            return self.result
        if player_id != self.current_player:
            raise ValueError(f"P{player_id + 1} cannot kong outside their turn")
        player = self.players[player_id]
        if player.hand.count(tile) < 4:
            raise ValueError(f"P{player_id + 1} cannot concealed-kong {tile_name(tile)}")
        robbed = self._resolve_rob_kong(player_id, tile, agents, allow=self.rules.allow_rob_concealed_kong)
        if robbed is not None:
            return robbed
        for _ in range(4):
            player.hand.remove(tile)
        player.melds.append(Meld("kong", [tile, tile, tile, tile], None, "concealed"))
        self.current_player = player_id
        self._pay_all_others(player_id, self.rules.concealed_kong_points_each, "concealed_kong", tile)
        self._log(f"P{player_id + 1} 暗槓 {tile_name(tile)}")
        return self._draw_replacement(player_id)

    def declare_added_kong(
        self,
        player_id: int,
        tile: int,
        meld_index: int,
        agents: list[ClaimAgent] | None = None,
    ) -> GameResult | None:
        if self.done:
            return self.result
        if player_id != self.current_player:
            raise ValueError(f"P{player_id + 1} cannot kong outside their turn")
        player = self.players[player_id]
        if tile not in player.hand:
            raise ValueError(f"P{player_id + 1} cannot add-kong without {tile_name(tile)}")
        if not (0 <= meld_index < len(player.melds)):
            raise ValueError(f"invalid meld index for added kong: {meld_index}")
        meld = player.melds[meld_index]
        if meld.kind != "pong" or meld.tiles[0] != tile:
            raise ValueError(f"P{player_id + 1} cannot add-kong meld {meld_index} with {tile_name(tile)}")

        robbed = self._resolve_rob_kong(player_id, tile, agents, allow=self.rules.allow_rob_added_kong)
        if robbed is not None:
            return robbed

        player.hand.remove(tile)
        meld.kind = "kong"
        meld.tiles = [tile, tile, tile, tile]
        meld.kong_type = "added"
        self.current_player = player_id
        self._pay_all_others(player_id, self.rules.added_kong_points_each, "added_kong", tile)
        self._log(f"P{player_id + 1} 加槓 {tile_name(tile)}")
        return self._draw_replacement(player_id)

    def _resolve_rob_kong(
        self,
        kong_player: int,
        tile: int,
        agents: list[ClaimAgent] | None,
        allow: bool,
    ) -> GameResult | None:
        if not allow:
            return None
        winners = []
        for offset in range(1, 4):
            pid = (kong_player + offset) % 4
            player = self.players[pid]
            if winning_with_tile(player.counts(), tile, len(player.melds)):
                winners.append(pid)
        if not winners:
            return None
        if not self.rules.allow_multi_win:
            winners = [winners[0]]
        return self._finish_rob_kong(winners, kong_player, tile)

    def _draw_replacement(self, player_id: int) -> GameResult | None:
        self.last_discard = None
        if not self.wall:
            return self._finish(None, None, "draw", "牌牆摸完，流局")
        tile = self.wall.pop()
        player = self.players[player_id]
        player.hand.append(tile)
        player.hand = sorted_tiles(player.hand)
        self.current_player = player_id
        self._log(f"P{player_id + 1} 槓上補牌")
        if self.can_self_win(player_id):
            return self.declare_self_win(player_id)
        return None

    def _draw_next(self, player_id: int) -> GameResult | None:
        self.last_discard = None
        self.current_player = player_id
        if not self.wall:
            return self._finish(None, None, "draw", "牌牆摸完，流局")
        tile = self.wall.pop()
        player = self.players[player_id]
        player.hand.append(tile)
        player.hand = sorted_tiles(player.hand)
        self._log(f"P{player_id + 1} 摸牌")
        return None

    def _finish_discard_win(self, winners: list[int], discarder: int, tile: int) -> GameResult:
        patterns = []
        for winner in winners:
            player = self.players[winner]
            counts = player.counts()
            counts[tile] += 1
            patterns.append(classify_win(counts, len(player.melds)))
            self._pay(discarder, winner, self.rules.ron_points, "discard_win", tile)
        names = ", ".join(f"P{winner + 1}" for winner in winners)
        message = f"{names} 食糊 {tile_name(tile)}"
        next_dealer = discarder if len(winners) > 1 else winners[0]
        return self._finish(
            winners[0],
            discarder,
            "multi_discard_win" if len(winners) > 1 else "discard_win",
            message,
            " / ".join(pattern for pattern in patterns if pattern),
            winners=winners,
            losers=[discarder],
            next_dealer=next_dealer,
        )

    def _finish_rob_kong(self, winners: list[int], kong_player: int, tile: int) -> GameResult:
        patterns = []
        package_points = self.rules.self_draw_points_each * 3
        for winner in winners:
            player = self.players[winner]
            counts = player.counts()
            counts[tile] += 1
            patterns.append(classify_win(counts, len(player.melds)))
            self._pay(kong_player, winner, package_points, "rob_kong", tile)
        names = ", ".join(f"P{winner + 1}" for winner in winners)
        message = f"{names} 搶槓 {tile_name(tile)}"
        return self._finish(
            winners[0],
            kong_player,
            "rob_kong",
            message,
            " / ".join(pattern for pattern in patterns if pattern),
            winners=winners,
            losers=[kong_player],
            next_dealer=kong_player if len(winners) > 1 else winners[0],
        )

    def _pay(self, payer: int, receiver: int, amount: int, kind: str, tile: int | None) -> None:
        if amount <= 0 or payer == receiver:
            return
        self.hand_score_delta[payer] -= amount
        self.hand_score_delta[receiver] += amount
        self.scores[payer] -= amount
        self.scores[receiver] += amount
        self.score_events.append(ScoreEvent(kind, receiver, payer, amount, tile))

    def _pay_all_others(self, receiver: int, amount_each: int, kind: str, tile: int | None) -> None:
        for payer in range(4):
            if payer != receiver:
                self._pay(payer, receiver, amount_each, kind, tile)

    def _finish(
        self,
        winner: int | None,
        loser: int | None,
        kind: str,
        message: str,
        pattern: str = "",
        winners: list[int] | None = None,
        losers: list[int] | None = None,
        next_dealer: int | None = None,
    ) -> GameResult:
        self.done = True
        if next_dealer is None:
            next_dealer = self.dealer if winner is None else winner
        self.dealer = next_dealer
        self.result = GameResult(
            winner,
            loser,
            kind,
            self.turns,
            len(self.wall),
            message,
            pattern,
            winners=list(winners or ([] if winner is None else [winner])),
            losers=list(losers or ([] if loser is None else [loser])),
            score_delta=list(self.hand_score_delta),
            next_dealer=next_dealer,
        )
        self._log(message)
        return self.result

    def _log(self, message: str) -> None:
        self.log.append(message)
