import asyncio
import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp
from thefuzz import process

from .models import MarketSnapshot, Player, PropLine

# Ingest resilience: timeout, retries, circuit breaker (env with defaults).
def _ingest_timeout_s() -> float:
    try:
        return float(os.getenv("INGEST_TIMEOUT_S", "12").strip())
    except (ValueError, TypeError):
        return 12.0


def _ingest_retries() -> int:
    try:
        return max(0, int(os.getenv("INGEST_RETRIES", "3").strip()))
    except (ValueError, TypeError):
        return 3


def _ingest_cb_max_failures() -> int:
    try:
        return max(1, int(os.getenv("INGEST_CB_MAX_FAILURES", "3").strip()))
    except (ValueError, TypeError):
        return 3


def _event_request_delay_s() -> float:
    """Delay between each event odds fetch (default 0.5s)."""
    try:
        return max(0.0, float(os.getenv("EVENT_REQUEST_DELAY_S", "0.5").strip()))
    except (ValueError, TypeError):
        return 0.5


def _ingest_429_backoff_base_s() -> float:
    """Base delay in seconds for 429 exponential backoff (default 2). Delays: base, 2*base, 4*base."""
    try:
        return max(0.5, float(os.getenv("INGEST_429_BACKOFF_BASE_S", "2").strip()))
    except (ValueError, TypeError):
        return 2.0


def _ingest_max_retries() -> int:
    """Max retries after initial attempt (default 3). Total attempts = 1 + this."""
    try:
        return max(0, int(os.getenv("INGEST_MAX_RETRIES", "3").strip()))
    except (ValueError, TypeError):
        return 3


# PropLine only accepts Points/Rebounds/Assists; filter to these when building lines.
ALLOWED_STAT_TYPES = ("Points", "Rebounds", "Assists")


def _stable_id(canonical_name: str) -> str:
    """Stable internal id: slug of canonical_name + short hash (deterministic)."""
    slug = re.sub(r"[^a-z0-9]+", "_", canonical_name.lower()).strip("_") or "player"
    h = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{h}"


# Maps provider-specific market keys into the canonical stat type labels used by PropLine.
STAT_TYPE_KEY_MAP: Dict[str, str] = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_points_rebounds_assists": "PRA",
    "player_threes": "Threes",
}


# Resolves noisy provider player names to canonical Player IDs using fuzzy string matching.
# Matches against Player.canonical_name and Player.aliases. Learns new players on the fly.
class FuzzyNameMatcher:
    def __init__(
        self,
        players: Iterable[Player],
        score_cutoff: int = 95,
        *,
        counters: Optional[Dict[str, int]] = None,
    ) -> None:
        self._score_cutoff = score_cutoff
        self._counters = counters  # Optional mutable dict for duplicate_player_alias_added, etc.
        # Maps canonical_name + aliases (and learned raw names) to Player.id
        self._name_to_player_id: Dict[str, str] = {}
        # Tracks Player objects by their id (including dynamically discovered ones)
        self._players_by_id: Dict[str, Player] = {}

        for player in players:
            self._players_by_id[player.id] = player
            self._name_to_player_id[player.canonical_name] = player.id
            for alias in player.aliases:
                self._name_to_player_id[alias] = player.id

        self._choices = list(self._name_to_player_id.keys())

    def _create_player_from_name(self, raw_provider_name: str) -> Player:
        """
        Create a new Player for a previously unseen name (no match cleared cutoff).
        provider_name = raw provider string; canonical_name = raw initially; nba_player_id = None.
        id = stable slug(canonical_name) + deterministic short hash. No collision suffix.
        """
        provider_name = raw_provider_name
        canonical_name = provider_name
        player_id = _stable_id(canonical_name)

        if player_id in self._players_by_id:
            self._name_to_player_id[raw_provider_name] = player_id
            return self._players_by_id[player_id]

        player = Player(
            id=player_id,
            provider_name=provider_name,
            canonical_name=canonical_name,
            nba_player_id=None,
            team="UNK",
            aliases=[],
        )
        self._players_by_id[player.id] = player
        self._name_to_player_id[raw_provider_name] = player.id
        self._choices = list(self._name_to_player_id.keys())
        return player

    def match_player_id(self, raw_provider_name: str) -> Optional[str]:
        """
        Returns the best-matching Player.id for a provider-supplied name.
        Matches against canonical_name and aliases only. If no match clears cutoff,
        dynamically creates a new Player (canonical_name = raw initially) and returns its id.
        """
        if not raw_provider_name:
            return None

        if not self._choices:
            return self._create_player_from_name(raw_provider_name).id

        match = process.extractOne(
            raw_provider_name, self._choices, score_cutoff=self._score_cutoff
        )
        if match is None:
            return self._create_player_from_name(raw_provider_name).id

        matched_name = match[0]
        existing_id = self._name_to_player_id.get(matched_name)
        if existing_id is None:
            return self._create_player_from_name(raw_provider_name).id
        if raw_provider_name not in self._name_to_player_id and self._counters is not None:
            self._counters["duplicate_player_alias_added"] = self._counters.get("duplicate_player_alias_added", 0) + 1
        self._name_to_player_id[raw_provider_name] = existing_id
        return existing_id

    def get_players_by_id(self) -> Dict[str, Player]:
        """Return all known players keyed by stable id (for resolution and display)."""
        return dict(self._players_by_id)


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


# Exponential backoff delays (seconds) for non-429 retries: 0.5, 1, 2.
_FETCH_BACKOFF = (0.5, 1.0, 2.0)


async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """
    HTTP GET with timeout (session default), retries (env INGEST_RETRIES / INGEST_MAX_RETRIES for 429).
    On HTTP 429: retry with exponential backoff (INGEST_429_BACKOFF_BASE_S * 2^attempt).
    Logs "429 rate limit; backing off {delay}s (attempt x/y)" on 429 retries.
    Raises after all attempts fail.
    """
    retries = _ingest_retries()
    max_retries_429 = _ingest_max_retries()
    base_429 = _ingest_429_backoff_base_s()
    total_429 = 1 + max_retries_429
    total_attempts = 1 + retries
    attempt_429 = 0
    attempt_generic = 0

    while True:
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    if attempt_429 < max_retries_429:
                        delay = base_429 * (2 ** attempt_429)
                        print(f"429 rate limit; backing off {delay:.1f}s (attempt {attempt_429 + 1}/{total_429})")
                        await asyncio.sleep(delay)
                        attempt_429 += 1
                        continue
                    response.raise_for_status()
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429 and attempt_429 < max_retries_429:
                delay = base_429 * (2 ** attempt_429)
                print(f"429 rate limit; backing off {delay:.1f}s (attempt {attempt_429 + 1}/{total_429})")
                await asyncio.sleep(delay)
                attempt_429 += 1
                continue
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            if attempt_generic < retries:
                delay = _FETCH_BACKOFF[attempt_generic] if attempt_generic < len(_FETCH_BACKOFF) else _FETCH_BACKOFF[-1]
                print(f"WARNING: Ingest fetch failed (attempt {attempt_generic + 1}/{total_attempts}), retrying in {delay}s: {e!r}")
                await asyncio.sleep(delay)
                attempt_generic += 1
                continue
            raise


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

        # Handle list responses, single-game dict responses, and generic dicts with "data" keys.
        if isinstance(payload, list):
            events = payload
        elif isinstance(payload, dict) and "bookmakers" in payload:
            # Single game from /events/{id}/odds
            events = [payload]
        elif isinstance(payload, dict):
            # Generic collection wrapper, e.g. {"data": [...]}
            events = payload.get("data", [])
        else:
            events = []

        for event in events:
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            bookmakers = event.get("bookmakers", [])
            for bookmaker in bookmakers:
                provider_label = bookmaker.get("title") or self.provider_name
                markets = bookmaker.get("markets", [])
                for market in markets:
                    market_key = market.get("key")
                    stat_type = STAT_TYPE_KEY_MAP.get(market_key)
                    if stat_type is None or stat_type not in ALLOWED_STAT_TYPES:
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

                    # Find the main line by calculating the tightest odds spread (juice)
                    main_lines = {}
                    for (player_name, threshold), odds_info in grouped.items():
                        over = odds_info["over_odds"]
                        under = odds_info["under_odds"]

                        if over is not None and under is not None:
                            o_prob = 100 / (over + 100) if over > 0 else abs(over) / (abs(over) + 100)
                            u_prob = 100 / (under + 100) if under > 0 else abs(under) / (abs(under) + 100)
                            imbalance = abs(o_prob - u_prob)
                        else:
                            imbalance = 999.0

                        if player_name not in main_lines or imbalance < main_lines[player_name][2]:
                            main_lines[player_name] = (threshold, odds_info, imbalance)

                    for player_name, (threshold, odds_info, imbalance) in main_lines.items():
                        # STRICT RULE: Drop one-way markets (alt-lines) entirely
                        if imbalance == 999.0:
                            continue

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
                                home_team=home_team,
                                away_team=away_team,
                            )
                            lines.append(line)
                        except ValueError:
                            continue

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
            if stat_type is None or stat_type not in ALLOWED_STAT_TYPES:
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


# Orchestrates ingestion from multiple providers into a single MarketSnapshot.
# Uses timeout, retries with backoff, and a per-bookmaker circuit breaker so failures do not crash the run.
# Returns (snapshot, players_by_id, counters). Partial snapshot if some providers fail.
async def fetch_multi_source_snapshot(
    snapshot_id: str,
    game_id: str,
    players: List[Player],
    providers: List[ProviderIngestor],
) -> Tuple[MarketSnapshot, Dict[str, Player], Dict[str, int]]:
    counters: Dict[str, int] = {}
    matcher = FuzzyNameMatcher(players, counters=counters)
    timeout_s = _ingest_timeout_s()
    cb_max = _ingest_cb_max_failures()
    # Per bookmaker (provider_name) failure count; after cb_max we skip that bookmaker for the rest of the run.
    circuit_failures: Dict[str, int] = {}

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    event_delay_s = _event_request_delay_s()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results: List[List[PropLine]] = []
        for i, provider in enumerate(providers):
            if i > 0 and event_delay_s > 0:
                await asyncio.sleep(event_delay_s)
            name = getattr(provider, "provider_name", "unknown")
            if circuit_failures.get(name, 0) >= cb_max:
                print(f"WARNING: Ingest skipping {name!r} (circuit breaker open after {cb_max} failures).")
                results.append([])
                continue
            try:
                lines = await provider.fetch_lines(session, matcher)
                results.append(lines if isinstance(lines, list) else [])
            except Exception as e:
                circuit_failures[name] = circuit_failures.get(name, 0) + 1
                if circuit_failures[name] >= cb_max:
                    print(f"WARNING: Ingest circuit breaker open for {name!r} after {cb_max} failures; skipping for remainder of run.")
                results.append([])

    all_lines: List[PropLine] = []
    for result in results:
        all_lines.extend(result)

    snapshot = MarketSnapshot(
        snapshot_id=snapshot_id,
        game_id=game_id,
        lines=all_lines,
    )
    return snapshot, matcher.get_players_by_id(), counters


def _profit_per_unit_american(odds: Optional[float]) -> float:
    """Profit per $1 stake if bet wins (American odds). Returns -1 if odds is None (worst case)."""
    if odds is None:
        return -1.0
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


# Stat types for which we aggregate multiple books (PTS/REB/AST and optionally others).
AGGREGATE_STAT_TYPES = ("Points", "Rebounds", "Assists")


def aggregate_snapshot_by_best_odds(
    snapshot: MarketSnapshot,
    stat_types: Optional[Tuple[str, ...]] = None,
) -> MarketSnapshot:
    """
    Group PropLines by (player_id, stat_type, line). For each group compute
    best_over_odds + over_provider and best_under_odds + under_provider across books;
    output one aggregated PropLine per group. If stat_types is set (e.g. PTS/REB/AST),
    only lines with that stat_type are included in the result; otherwise all.
    """
    key_type = Tuple[str, str, float]  # (player_id, stat_type, threshold)
    allowed = set(stat_types) if stat_types is not None else None
    grouped: Dict[key_type, List[PropLine]] = {}

    for line in snapshot.lines:
        if allowed is not None and line.stat_type not in allowed:
            continue
        key: key_type = (line.player_id, line.stat_type, line.threshold)
        grouped.setdefault(key, []).append(line)

    aggregated_lines: List[PropLine] = []
    for (player_id, stat_type, threshold), lines in grouped.items():
        best_over = max(
            (l for l in lines if l.over_odds is not None),
            key=lambda l: _profit_per_unit_american(l.over_odds),
            default=None,
        )
        best_under = max(
            (l for l in lines if l.under_odds is not None),
            key=lambda l: _profit_per_unit_american(l.under_odds),
            default=None,
        )
        over_odds = best_over.over_odds if best_over else None
        under_odds = best_under.under_odds if best_under else None
        over_provider = best_over.provider if best_over else None
        under_provider = best_under.provider if best_under else None
        primary = over_provider or under_provider or (lines[0].provider if lines else "aggregated")

        first = lines[0]
        aggregated_lines.append(
            PropLine(
                player_id=player_id,
                provider=primary,
                stat_type=stat_type,
                threshold=threshold,
                over_odds=over_odds,
                under_odds=under_odds,
                over_provider=over_provider,
                under_provider=under_provider,
                home_team=getattr(first, "home_team", None),
                away_team=getattr(first, "away_team", None),
            )
        )

    return MarketSnapshot(
        snapshot_id=snapshot.snapshot_id,
        timestamp=snapshot.timestamp,
        game_id=snapshot.game_id,
        lines=aggregated_lines,
    )

