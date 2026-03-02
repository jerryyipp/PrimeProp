"""
Historical stats client and simple 24-hour cache for player game logs.
Uses the official nba_api package to fetch the last N games directly from NBA.com.
Results are cached on disk for 24 hours to avoid repeatedly hitting the API.

Main.py contract:
  - resolve_nba_player_id(canonical_name: str) -> Optional[int]
  - async fetch_last_n_games(nba_player_id: int, n: int) -> list[dict]  # GAME_DATE, PTS, REB, AST
  - get_stat_series(games: list[dict], stat_type: str) -> list[float]   # Points, Rebounds, Assists
"""

__all__ = ["resolve_nba_player_id", "fetch_last_n_games", "get_stat_series"]

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

from .projection import StatType

_CACHE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = _CACHE_DIR / "stats_cache.json"
NBA_ID_CACHE_PATH = _CACHE_DIR / "nba_player_id_cache.json"
GAME_RESULT_CACHE_PATH = _CACHE_DIR / "game_result_cache.json"
CACHE_TTL = timedelta(hours=24)
GAME_RESULT_CACHE_TTL = timedelta(days=7)

# In-memory cache for canonical_name -> nba_player_id (persisted to disk on resolve)
_nba_id_memory_cache: Dict[str, int] = {}


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


def _load_nba_id_cache() -> Dict[str, int]:
    """Load canonical_name -> nba_player_id from disk (no TTL; ids are stable)."""
    if not NBA_ID_CACHE_PATH.exists():
        return {}
    try:
        with NBA_ID_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return {}


def _save_nba_id_cache(cache: Dict[str, int]) -> None:
    try:
        with NBA_ID_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        pass


def resolve_nba_player_id(canonical_name: str) -> Optional[int]:
    """
    Resolve canonical_name -> nba_player_id. Uses in-memory then disk cache (nba_player_id_cache.json),
    then nba_api find_players_by_full_name. Returns None if resolution fails (caller should skip).
    """
    if not canonical_name or not canonical_name.strip():
        return None
    canonical_name = canonical_name.strip()

    global _nba_id_memory_cache
    if not _nba_id_memory_cache:
        _nba_id_memory_cache = _load_nba_id_cache()

    if canonical_name in _nba_id_memory_cache:
        return _nba_id_memory_cache[canonical_name]

    nba_players = players.find_players_by_full_name(canonical_name)
    if not nba_players:
        return None
    nba_id = int(nba_players[0]["id"])
    _nba_id_memory_cache[canonical_name] = nba_id
    _save_nba_id_cache(_nba_id_memory_cache)
    return nba_id


def _cache_key(player_name_or_id: str, stat_type: StatType, n_games: int) -> str:
    """Cache key for game logs; use canonical_name or str(nba_player_id)."""
    return f"{player_name_or_id}|{stat_type}|{n_games}"


def _gamelog_cache_key(nba_player_id: int, n: int) -> str:
    """Cache key for raw gamelog (last n games)."""
    return f"gamelog|{nba_player_id}|{n}"


def _get_cached_gamelog(nba_player_id: int, n: int) -> Optional[List[Dict]]:
    """Return cached gamelog rows if present and not expired."""
    cache = _load_cache()
    key = _gamelog_cache_key(nba_player_id, n)
    entry = cache.get(key)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(entry["cached_at"])
    except Exception:
        return None
    if datetime.now(timezone.utc) - ts > CACHE_TTL:
        return None
    rows = entry.get("games")
    if not isinstance(rows, list):
        return None
    return rows


def _set_cached_gamelog(nba_player_id: int, n: int, games: List[Dict]) -> None:
    cache = _load_cache()
    key = _gamelog_cache_key(nba_player_id, n)
    cache[key] = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "games": games,
    }
    _save_cache(cache)


def _fetch_gamelog_sync(nba_player_id: int, n: int) -> List[Dict]:
    """Fetch last n games from nba_api; return list of dicts with GAME_DATE, PTS, REB, AST."""
    try:
        log = playergamelog.PlayerGameLog(player_id=nba_player_id)
        df = log.get_data_frames()[0]
    except Exception:
        return []
    if df is None or df.empty:
        return []
    df = df.head(n).iloc[::-1]  # oldest -> newest (chronological order for main.py)
    rows: List[Dict] = []
    for _, row in df.iterrows():
        try:
            gd = row.get("GAME_DATE")
            rows.append({
                "GAME_DATE": str(gd) if gd is not None else "",
                "PTS": float(row["PTS"]),
                "REB": float(row["REB"]),
                "AST": float(row["AST"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return rows


async def fetch_last_n_games(nba_player_id: int, n: int) -> List[Dict]:
    """
    Fetch last n games for a player; each row has GAME_DATE, PTS, REB, AST, FG3M.
    Uses existing cache folder + TTL. Prefer this over per-stat fetches when building multiple series.
    """
    cached = _get_cached_gamelog(nba_player_id, n)
    if cached is not None:
        return cached
    await asyncio.sleep(0.5)
    games = await asyncio.to_thread(_fetch_gamelog_sync, nba_player_id, n)
    if games:
        _set_cached_gamelog(nba_player_id, n, games)
    return games


def get_stat_series(games: List[Dict], stat_type: str) -> List[float]:
    """
    Extract a stat series from gamelog rows. Supports only Points, Rebounds, Assists.
    stat_type: 'Points'|'Rebounds'|'Assists' or lowercase/aliases (points, rebounds, assists, pts, reb, ast).
    Returns list of floats (oldest to newest).
    """
    if not games:
        return []
    st = (stat_type or "").strip().lower()
    if st in ("points", "pts"):
        return [float(g["PTS"]) for g in games if "PTS" in g]
    if st in ("rebounds", "reb"):
        return [float(g["REB"]) for g in games if "REB" in g]
    if st in ("assists", "ast"):
        return [float(g["AST"]) for g in games if "AST" in g]
    return []


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


def _fetch_from_nba_api_sync(
    canonical_name: str,
    stat_type: StatType,
    n_games: int,
    nba_player_id: Optional[int] = None,
) -> List[float]:
    """Synchronous worker that hits NBA.com. Prefers nba_player_id when provided."""
    if nba_player_id is not None:
        nba_id = nba_player_id
    else:
        nba_players = players.find_players_by_full_name(canonical_name)
        if not nba_players:
            return []
        nba_id = nba_players[0]["id"]

    try:
        log = playergamelog.PlayerGameLog(player_id=nba_id)
        df = log.get_data_frames()[0]
    except Exception as e:
        print(f"DEBUG: NBA API error for {canonical_name} (id={nba_id}): {e}")
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


def _game_result_cache_key(player_nba_id: int, game_date: date, stat_type: StatType) -> str:
    return f"{player_nba_id}|{game_date.isoformat()}|{stat_type}"


def _load_game_result_cache() -> Dict[str, Dict]:
    if not GAME_RESULT_CACHE_PATH.exists():
        return {}
    try:
        with GAME_RESULT_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_game_result_cache(cache: Dict[str, Dict]) -> None:
    try:
        with GAME_RESULT_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        pass


def _get_cached_game_result(
    player_nba_id: int, game_date: date, stat_type: StatType
) -> Optional[float]:
    cache = _load_game_result_cache()
    key = _game_result_cache_key(player_nba_id, game_date, stat_type)
    entry = cache.get(key)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(entry["cached_at"])
    except Exception:
        return None
    if datetime.now(timezone.utc) - ts > GAME_RESULT_CACHE_TTL:
        return None
    val = entry.get("value")
    if val is None:
        return None
    return float(val)


def _set_cached_game_result(
    player_nba_id: int, game_date: date, stat_type: StatType, value: float
) -> None:
    cache = _load_game_result_cache()
    key = _game_result_cache_key(player_nba_id, game_date, stat_type)
    cache[key] = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "value": value,
    }
    _save_game_result_cache(cache)


def _parse_game_date(date_str: str) -> Optional[date]:
    """Parse nba_api GAME_DATE ('OCT 26, 2023' or '2024-10-22') to date."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    try:
        return datetime.strptime(date_str, "%b %d, %Y").date()
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None


def fetch_game_result(
    player_nba_id: int,
    game_date: Union[date, datetime],
    stat_type: StatType,
) -> Optional[float]:
    """
    Fetch actual PTS/REB/AST (or PRA/Threes) for a player on a game date using nba_api PlayerGameLog.
    If no exact date match, returns the value from the closest matching game (by date).
    Returns None if no game found. Results are cached (7-day TTL).
    """
    if isinstance(game_date, datetime):
        game_date = game_date.date()

    cached = _get_cached_game_result(player_nba_id, game_date, stat_type)
    if cached is not None:
        return cached

    try:
        log = playergamelog.PlayerGameLog(player_id=player_nba_id)
        df = log.get_data_frames()[0]
    except Exception:
        return None
    if df is None or df.empty:
        return None

    # Build list of (parsed_date, stat_value); find exact or closest match
    candidates: List[tuple[date, float]] = []
    for _, row in df.iterrows():
        try:
            parsed = _parse_game_date(row.get("GAME_DATE"))
            if parsed is None:
                continue
            if stat_type == "Points":
                val = float(row["PTS"])
            elif stat_type == "Rebounds":
                val = float(row["REB"])
            elif stat_type == "Assists":
                val = float(row["AST"])
            elif stat_type == "PRA":
                val = float(row["PTS"] + row["REB"] + row["AST"])
            elif stat_type == "Threes":
                val = float(row["FG3M"])
            else:
                continue
            candidates.append((parsed, val))
        except (ValueError, KeyError, TypeError):
            continue

    if not candidates:
        return None

    # Exact match first
    for d, val in candidates:
        if d == game_date:
            _set_cached_game_result(player_nba_id, game_date, stat_type, val)
            return val

    # Closest matching game by date distance (within 7 days)
    def days_diff(d: date) -> int:
        return abs((d - game_date).days)

    best = min(candidates, key=lambda x: days_diff(x[0]))
    if days_diff(best[0]) > 7:
        return None
    val = best[1]
    _set_cached_game_result(player_nba_id, game_date, stat_type, val)
    return val


async def fetch_last_n_game_values(
    canonical_name: str,
    stat_type: StatType,
    n_games: int = 10,
    nba_player_id: Optional[int] = None,
    session=None,  # Kept for compatibility with main.py calls
) -> List[float]:
    """
    Fetch last N game values using the nba_api package.
    Prefers nba_player_id when provided; otherwise resolves by canonical_name.
    Cache key uses nba_player_id when available so lookups are stable.
    """
    lookup_key = str(nba_player_id) if nba_player_id is not None else canonical_name
    cached = _get_cached_values(lookup_key, stat_type, n_games)
    if cached is not None:
        return cached

    await asyncio.sleep(0.5)
    values = await asyncio.to_thread(
        _fetch_from_nba_api_sync,
        canonical_name,
        stat_type,
        n_games,
        nba_player_id,
    )
    if values:
        _set_cached_values(lookup_key, stat_type, n_games, values)
    return values
