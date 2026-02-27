"""
Historical stats client and simple 24-hour cache for player game logs.

Uses a free public NBA stats API (balldontlie.io) to fetch the last N games
for a given player and stat type. Results are cached on disk for 24 hours to
avoid repeatedly hitting the API for the same player/stat combination.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

CACHE_PATH = Path(__file__).resolve().parent.parent / "stats_cache.json"
CACHE_TTL = timedelta(hours=24)

_BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")
_HEADERS = {"Authorization": f"Bearer {_BALLDONTLIE_API_KEY}"} if _BALLDONTLIE_API_KEY else {}

# Map our stat types to balldontlie fields.
_STAT_FIELD_MAP: Dict[StatType, Optional[str]] = {
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
    "PRA": None,  # computed as pts + reb + ast
    "Threes": "fg3m",
}


def _load_cache() -> Dict[str, Dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        # Cache failures should never break the main pipeline.
        pass


def _cache_key(player_name: str, stat_type: StatType, n_games: int) -> str:
    return f"{player_name}|{stat_type}|{n_games}"


def _get_cached_values(player_name: str, stat_type: StatType, n_games: int) -> Optional[List[float]]:
    cache = _load_cache()
    key = _cache_key(player_name, stat_type, n_games)
    entry = cache.get(key)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(entry["cached_at"])
    except Exception:
        return None
    if datetime.now(timezone.utc) - ts > CACHE_TTL:
        return None
    values = entry.get("values")
    if not isinstance(values, list):
        return None
    return [float(v) for v in values]


def _set_cached_values(player_name: str, stat_type: StatType, n_games: int, values: List[float]) -> None:
    cache = _load_cache()
    key = _cache_key(player_name, stat_type, n_games)
    cache[key] = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "values": values,
    }
    _save_cache(cache)


async def _fetch_player_id(session: aiohttp.ClientSession, player_name: str) -> Optional[int]:
    """
    Resolve a player name to a balldontlie player id using the search endpoint.
    """
    url = "https://api.balldontlie.io/v1/players"
    params = {"search": player_name}
    async with session.get(url, params=params, headers=_HEADERS) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
        items = data.get("data") or []
        if not items:
            return None
        # Take the first match; in practice you may want league/team filters.
        return items[0].get("id")


async def _fetch_last_n_games_raw(
    session: aiohttp.ClientSession,
    player_id: int,
    n_games: int,
) -> List[Dict]:
    """
    Fetch raw game log stats for a player and return the most recent N games.
    """
    url = "https://api.balldontlie.io/v1/stats"
    params = {
        "player_ids[]": player_id,
        "per_page": n_games,
        "postseason": "false",
    }
    async with session.get(url, params=params, headers=_HEADERS) as resp:
        if resp.status != 200:
            return []
        data = await resp.json()
        items = data.get("data") or []
        # Ensure oldest -> newest order based on game date.
        items.sort(key=lambda g: g.get("game", {}).get("date", ""))
        return items[-n_games:]


def _extract_stat_values(games: List[Dict], stat_type: StatType) -> List[float]:
    if not games:
        return []
    field = _STAT_FIELD_MAP[stat_type]
    values: List[float] = []
    for g in games:
        stats = g
        try:
            if stat_type == "PRA":
                pts = float(stats.get("pts", 0.0))
                reb = float(stats.get("reb", 0.0))
                ast = float(stats.get("ast", 0.0))
                values.append(pts + reb + ast)
            else:
                if field is None:
                    continue
                raw = stats.get(field)
                if raw is None:
                    continue
                values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


async def fetch_last_n_game_values(
    player_name: str,
    stat_type: StatType,
    n_games: int = 10,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[float]:
    """
    Fetch last N game values for (player_name, stat_type).

    Uses a 24-hour on-disk cache keyed by (player_name, stat_type, n_games).
    Returns an ordered list [oldest, ..., newest]; may be shorter than N if
    not enough games are available, or empty if the player cannot be found.
    """
    cached = _get_cached_values(player_name, stat_type, n_games)
    if cached is not None:
        return cached

    owns_session = False
    if session is None:
        session = aiohttp.ClientSession()
        owns_session = True

    try:
        player_id = await _fetch_player_id(session, player_name)
        if player_id is None:
            return []

        games = await _fetch_last_n_games_raw(session, player_id, n_games)
        values = _extract_stat_values(games, stat_type)
        if values:
            _set_cached_values(player_name, stat_type, n_games, values)
        return values
    finally:
        if owns_session and session is not None:
            await session.close()

