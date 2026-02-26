import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp
from thefuzz import process

from .models import MarketSnapshot, Player, PropLine


# Maps provider-specific market keys into the canonical stat type labels used by PropLine.
STAT_TYPE_KEY_MAP: Dict[str, str] = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_points_rebounds_assists": "PRA",
    "player_threes": "Threes",
}


# Resolves noisy provider player names to canonical Player IDs using fuzzy string matching.
class FuzzyNameMatcher:
    def __init__(self, players: Iterable[Player], score_cutoff: int = 80) -> None:
        self._score_cutoff = score_cutoff
        self._name_to_player_id: Dict[str, str] = {}

        for player in players:
            self._name_to_player_id[player.standardized_name] = player.id
            for alias in player.aliases:
                self._name_to_player_id[alias] = player.id

        self._choices: List[str] = list(self._name_to_player_id.keys())

    # Returns the best-matching Player.id for a provider-supplied name, or None if nothing clears the score cutoff.
    def match_player_id(self, name: str) -> Optional[str]:
        if not name or not self._choices:
            return None

        match = process.extractOne(name, self._choices, score_cutoff=self._score_cutoff)
        if match is None:
            return None

        matched_name = match[0]
        return self._name_to_player_id.get(matched_name)


# Common interface for all upstream data sources that can emit normalized PropLine objects.
class ProviderIngestor(ABC):
    provider_name: str

    @abstractmethod
    async def fetch_lines(
        self,
        session: aiohttp.ClientSession,
        matcher: FuzzyNameMatcher,
    ) -> List[PropLine]:
        raise NotImplementedError


# Small helper for issuing an HTTP GET and decoding JSON, letting HTTP errors surface naturally.
async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    async with session.get(url, params=params, headers=headers) as response:
        response.raise_for_status()
        return await response.json()


# Ingestor for The Odds API–style payloads; normalizes markets/outcomes into PropLine instances.
class OddsApiIngestor(ProviderIngestor):
    def __init__(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.url = url
        self.params = params or {}
        self.headers = headers or {}
        self.provider_name = "The Odds API"

    async def fetch_lines(
        self,
        session: aiohttp.ClientSession,
        matcher: FuzzyNameMatcher,
    ) -> List[PropLine]:
        payload = await _fetch_json(session, self.url, params=self.params, headers=self.headers)
        return self._parse_payload(payload, matcher)

    # Transforms a raw The Odds API payload into PropLine objects, grouping over/under odds by player and threshold.
    def _parse_payload(self, payload: Any, matcher: FuzzyNameMatcher) -> List[PropLine]:
        lines: List[PropLine] = []
        events = payload if isinstance(payload, list) else payload.get("data", [])

        for event in events:
            bookmakers = event.get("bookmakers", [])
            for bookmaker in bookmakers:
                provider_label = bookmaker.get("title") or self.provider_name
                markets = bookmaker.get("markets", [])
                for market in markets:
                    market_key = market.get("key")
                    stat_type = STAT_TYPE_KEY_MAP.get(market_key)
                    if stat_type is None:
                        continue

                    outcomes = market.get("outcomes", [])
                    grouped: Dict[Tuple[str, float], Dict[str, Optional[int]]] = {}

                    for outcome in outcomes:
                        player_name = outcome.get("description") or outcome.get("player")
                        if not player_name:
                            continue

                        threshold_raw = outcome.get("point")
                        if threshold_raw is None:
                            continue

                        try:
                            threshold = float(threshold_raw)
                        except (TypeError, ValueError):
                            continue

                        name_key = (player_name, threshold)
                        price = outcome.get("price")
                        outcome_name = str(outcome.get("name") or "").lower()

                        info = grouped.setdefault(name_key, {"over_odds": None, "under_odds": None})
                        if outcome_name == "over":
                            info["over_odds"] = price
                        elif outcome_name == "under":
                            info["under_odds"] = price

                    for (player_name, threshold), odds_info in grouped.items():
                        player_id = matcher.match_player_id(player_name)
                        if player_id is None:
                            continue

                        try:
                            line = PropLine(
                                player_id=player_id,
                                provider=provider_label,
                                stat_type=stat_type,  # type: ignore[arg-type]
                                threshold=threshold,
                                over_odds=odds_info["over_odds"],
                                under_odds=odds_info["under_odds"],
                            )
                        except ValueError:
                            continue

                        lines.append(line)

        return lines


# Maps various PrizePicks stat labels into our canonical stat type names.
STAT_TYPE_NAME_MAP: Dict[str, str] = {
    "points": "Points",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "points_rebounds_assists": "PRA",
    "pra": "PRA",
    "threes": "Threes",
    "three_pointers_made": "Threes",
}


# Ingestor for PrizePicks-style projection data; converts entries into normalized PropLine objects.
class PrizePicksIngestor(ProviderIngestor):
    def __init__(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.url = url
        self.params = params or {}
        self.headers = headers or {}
        self.provider_name = "PrizePicks"

    async def fetch_lines(
        self,
        session: aiohttp.ClientSession,
        matcher: FuzzyNameMatcher,
    ) -> List[PropLine]:
        payload = await _fetch_json(session, self.url, params=self.params, headers=self.headers)
        return self._parse_payload(payload, matcher)

    # Converts a PrizePicks-style JSON payload into PropLine records, ignoring projections we cannot normalize safely.
    def _parse_payload(self, payload: Any, matcher: FuzzyNameMatcher) -> List[PropLine]:
        lines: List[PropLine] = []
        items = payload.get("data", []) if isinstance(payload, dict) else payload

        for item in items:
            attributes = item.get("attributes", {})
            player_name = attributes.get("display_name") or attributes.get("name")
            if not player_name:
                continue

            stat_raw = attributes.get("stat_type") or attributes.get("stat")
            if not stat_raw:
                continue

            stat_key = str(stat_raw).lower()
            stat_type = STAT_TYPE_NAME_MAP.get(stat_key)
            if stat_type is None:
                continue

            line_score = attributes.get("line_score")
            if line_score is None:
                continue

            try:
                threshold = float(line_score)
            except (TypeError, ValueError):
                continue

            player_id = matcher.match_player_id(player_name)
            if player_id is None:
                continue

            try:
                line = PropLine(
                    player_id=player_id,
                    provider=self.provider_name,
                    stat_type=stat_type,  # type: ignore[arg-type]
                    threshold=threshold,
                    over_odds=None,
                    under_odds=None,
                )
            except ValueError:
                continue

            lines.append(line)

        return lines


# Orchestrates concurrent ingestion from multiple providers into a single MarketSnapshot.
async def fetch_multi_source_snapshot(
    snapshot_id: str,
    game_id: str,
    players: List[Player],
    providers: List[ProviderIngestor],
) -> MarketSnapshot:
    matcher = FuzzyNameMatcher(players)

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(provider.fetch_lines(session, matcher) for provider in providers),
            return_exceptions=True,
        )

    all_lines: List[PropLine] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        all_lines.extend(result)

    return MarketSnapshot(
        snapshot_id=snapshot_id,
        game_id=game_id,
        lines=all_lines,
    )

