"""
Historical stats client and simple 24-hour cache for player game logs.
Uses the official nba_api package to fetch the last N games directly from NBA.com.
Results are cached on disk for 24 hours to avoid repeatedly hitting the API.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

from src.projection import StatType

CACHE_PATH = Path(__file__).resolve().parent.parent / "stats_cache.json"
CACHE_TTL = timedelta(hours=24)


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


def _fetch_from_nba_api_sync(player_name: str, stat_type: StatType, n_games: int) -> List[float]:
    """Synchronous worker that hits NBA.com."""
    nba_players = players.find_players_by_full_name(player_name)
    if not nba_players:
        return []

    player_id = nba_players[0]["id"]
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id)
        df = log.get_data_frames()[0]
    except Exception as e:
        print(f"DEBUG: NBA API error for {player_name}: {e}")
        return []
    if df.empty:
        return []
    # NBA.com returns newest games first. We want the last n_games, ordered oldest -> newest
    df = df.head(n_games).iloc[::-1]
    values: List[float] = []
    for _, row in df.iterrows():
        try:
            if stat_type == "Points":
                values.append(float(row["PTS"]))
            elif stat_type == "Rebounds":
                values.append(float(row["REB"]))
            elif stat_type == "Assists":
                values.append(float(row["AST"]))
            elif stat_type == "PRA":
                values.append(float(row["PTS"] + row["REB"] + row["AST"]))
            elif stat_type == "Threes":
                values.append(float(row["FG3M"]))
        except (KeyError, ValueError, TypeError):
            continue
    return values


async def fetch_last_n_game_values(
    player_name: str,
    stat_type: StatType,
    n_games: int = 10,
    session=None,  # Kept for compatibility with main.py calls
) -> List[float]:
    """
    Fetch last N game values using the nba_api package.
    """
    cached = _get_cached_values(player_name, stat_type, n_games)
    if cached is not None:
        return cached

    # A tiny 0.5s delay to safely navigate NBA.com's spam filters
    await asyncio.sleep(0.5)
    # Run the synchronous nba_api network calls safely in a background thread
    values = await asyncio.to_thread(_fetch_from_nba_api_sync, player_name, stat_type, n_games)
    if values:
        _set_cached_values(player_name, stat_type, n_games, values)
    return values
