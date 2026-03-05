"""
Opponent/matchup adjustment: team metrics cache and player team resolution.

Provides daily-cached per-team defensive metric (points allowed per game, true
opponent metric) and league average for projection adjustment. get_opponent_points_factor(team)
returns a multiplier around 1.0 for scoring adjustment.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_CACHE_DIR = Path(__file__).resolve().parent.parent
TEAM_METRICS_CACHE_PATH = _CACHE_DIR / "team_metrics_cache.json"
CACHE_TTL = timedelta(hours=24)

# League-average defensive metrics (fallback when API unavailable).
DEFAULT_LEAGUE_PTS_ALLOWED = 112.0
DEFAULT_LEAGUE_REB_ALLOWED = 44.0
DEFAULT_LEAGUE_AST_ALLOWED = 25.0

# Map Odds API / display names (and abbrevs) to canonical NBA abbrev for cache lookup.
TEAM_NAME_TO_ABBREV: Dict[str, str] = {
    "ATL": "ATL", "ATLANTA": "ATL", "HAWKS": "ATL", "ATLANTA HAWKS": "ATL",
    "BOS": "BOS", "BOSTON": "BOS", "CELTICS": "BOS", "BOSTON CELTICS": "BOS",
    "BKN": "BKN", "BROOKLYN": "BKN", "NETS": "BKN", "BROOKLYN NETS": "BKN",
    "CHA": "CHA", "CHARLOTTE": "CHA", "HORNETS": "CHA", "CHARLOTTE HORNETS": "CHA",
    "CHI": "CHI", "CHICAGO": "CHI", "BULLS": "CHI", "CHICAGO BULLS": "CHI",
    "CLE": "CLE", "CLEVELAND": "CLE", "CAVALIERS": "CLE", "CAVS": "CLE", "CLEVELAND CAVALIERS": "CLE",
    "DAL": "DAL", "DALLAS": "DAL", "MAVERICKS": "DAL", "MAVS": "DAL", "DALLAS MAVERICKS": "DAL",
    "DEN": "DEN", "DENVER": "DEN", "NUGGETS": "DEN", "DENVER NUGGETS": "DEN",
    "DET": "DET", "DETROIT": "DET", "PISTONS": "DET", "DETROIT PISTONS": "DET",
    "GSW": "GSW", "GOLDEN STATE": "GSW", "WARRIORS": "GSW", "GOLDEN STATE WARRIORS": "GSW", "GS": "GSW",
    "HOU": "HOU", "HOUSTON": "HOU", "ROCKETS": "HOU", "HOUSTON ROCKETS": "HOU",
    "IND": "IND", "INDIANA": "IND", "PACERS": "IND", "INDIANA PACERS": "IND",
    "LAC": "LAC", "LA CLIPPERS": "LAC", "CLIPPERS": "LAC", "LOS ANGELES CLIPPERS": "LAC",
    "LAL": "LAL", "LAKERS": "LAL", "LOS ANGELES LAKERS": "LAL", "LA LAKERS": "LAL",
    "MEM": "MEM", "MEMPHIS": "MEM", "GRIZZLIES": "MEM", "MEMPHIS GRIZZLIES": "MEM",
    "MIA": "MIA", "MIAMI": "MIA", "HEAT": "MIA", "MIAMI HEAT": "MIA",
    "MIL": "MIL", "MILWAUKEE": "MIL", "BUCKS": "MIL", "MILWAUKEE BUCKS": "MIL",
    "MIN": "MIN", "MINNESOTA": "MIN", "TIMBERWOLVES": "MIN", "WOLVES": "MIN", "MINNESOTA TIMBERWOLVES": "MIN",
    "NOP": "NOP", "NEW ORLEANS": "NOP", "PELICANS": "NOP", "NEW ORLEANS PELICANS": "NOP", "NO": "NOP",
    "NYK": "NYK", "NEW YORK": "NYK", "KNICKS": "NYK", "NEW YORK KNICKS": "NYK", "NY": "NYK",
    "OKC": "OKC", "OKLAHOMA CITY": "OKC", "THUNDER": "OKC", "OKLAHOMA CITY THUNDER": "OKC",
    "ORL": "ORL", "ORLANDO": "ORL", "MAGIC": "ORL", "ORLANDO MAGIC": "ORL",
    "PHI": "PHI", "PHILADELPHIA": "PHI", "76ERS": "PHI", "SIXERS": "PHI", "PHILADELPHIA 76ERS": "PHI",
    "PHX": "PHX", "PHOENIX": "PHX", "SUNS": "PHX", "PHOENIX SUNS": "PHX",
    "POR": "POR", "PORTLAND": "POR", "TRAIL BLAZERS": "POR", "BLAZERS": "POR", "PORTLAND TRAIL BLAZERS": "POR",
    "SAC": "SAC", "SACRAMENTO": "SAC", "KINGS": "SAC", "SACRAMENTO KINGS": "SAC",
    "SAS": "SAS", "SAN ANTONIO": "SAS", "SPURS": "SAS", "SAN ANTONIO SPURS": "SAS", "SA": "SAS",
    "TOR": "TOR", "TORONTO": "TOR", "RAPTORS": "TOR", "TORONTO RAPTORS": "TOR",
    "UTA": "UTA", "UTAH": "UTA", "JAZZ": "UTA", "UTAH JAZZ": "UTA",
    "WAS": "WAS", "WASHINGTON": "WAS", "WIZARDS": "WAS", "WASHINGTON WIZARDS": "WAS", "WSH": "WAS",
}


def normalize_team_to_abbrev(team_name_or_abbrev: str) -> str:
    """Return canonical NBA abbreviation for cache lookup. Unknown names returned uppercased as-is."""
    if not team_name_or_abbrev or not team_name_or_abbrev.strip():
        return ""
    raw = team_name_or_abbrev.strip().upper()
    return TEAM_NAME_TO_ABBREV.get(raw, raw)


def _load_cache() -> Dict[str, Any]:
    if not TEAM_METRICS_CACHE_PATH.exists():
        return {}
    try:
        with TEAM_METRICS_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(data: Dict[str, Any]) -> None:
    try:
        with TEAM_METRICS_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


def _is_expired(entry: Dict[str, Any]) -> bool:
    try:
        ts = datetime.fromisoformat(entry.get("cached_at", ""))
        return datetime.now(timezone.utc) - ts > CACHE_TTL
    except Exception:
        return True


def _ensure_team_metrics_cache() -> None:
    """Populate cache with per-team points allowed (and league avg) if missing or expired."""
    cache = _load_cache()
    league_entry = cache.get("league_avg")
    if league_entry is not None and not _is_expired(league_entry):
        return
    _fetch_and_cache_daily_pts_allowed()


_logged_matchup_unavailable = False


def _log_matchup_unavailable_once() -> None:
    global _logged_matchup_unavailable
    if not _logged_matchup_unavailable:
        _logged_matchup_unavailable = True
        print("Matchup metrics unavailable; using neutral adjustment")


def _fetch_and_cache_daily_pts_allowed() -> None:
    """
    Fetch per-team metrics from NBA API LeagueDashTeamStats (PerGame).
    Uses defensively relevant columns if available; otherwise caches neutral metrics so factor=1.0.
    Caching and TTL unchanged. Never raises: on failure returns and callers use fallback dicts.
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats

        obj = leaguedashteamstats.LeagueDashTeamStats(per_mode_detailed="PerGame")
        dfs = obj.get_data_frames()
        if not dfs or dfs[0] is None or dfs[0].empty:
            _log_matchup_unavailable_once()
            return
        df = dfs[0]
        abbrev_col = None
        for c in ("TEAM_ABBREVIATION", "ABBREVIATION", "TEAM_NAME"):
            if c in df.columns:
                abbrev_col = c
                break
        if abbrev_col is None:
            _log_matchup_unavailable_once()
            return

        # LeagueDashTeamStats gives team's own PTS/REB/AST, not "allowed". Look for defensive-style columns if any.
        defensively_relevant = None
        for col in ("OPP_PTS", "PTS_ALLOWED", "OPP_PTS_PER_GAME"):
            if col in df.columns:
                defensively_relevant = col
                break
        # If no defensive column, use neutral: cache league avg for all teams so factor = 1.0
        if defensively_relevant is None:
            _log_matchup_unavailable_once()
            now = datetime.now(timezone.utc)
            neutral = {
                "pts_allowed_per_game": DEFAULT_LEAGUE_PTS_ALLOWED,
                "reb_allowed_per_game": DEFAULT_LEAGUE_REB_ALLOWED,
                "ast_allowed_per_game": DEFAULT_LEAGUE_AST_ALLOWED,
            }
            cache = _load_cache()
            cache["league_avg"] = {"cached_at": now.isoformat(), "metrics": dict(neutral)}
            cache["teams"] = {}
            for _, row in df.iterrows():
                abbr = str(row[abbrev_col]).strip().upper()
                if abbr:
                    cache["teams"][abbr] = {"cached_at": now.isoformat(), "metrics": dict(neutral)}
            _save_cache(cache)
            return

        has_reb = "REB" in df.columns or "OPP_REB" in df.columns
        has_ast = "AST" in df.columns or "OPP_AST" in df.columns
        pts_col = defensively_relevant
        reb_col = "OPP_REB" if "OPP_REB" in df.columns else ("REB" if has_reb else None)
        ast_col = "OPP_AST" if "OPP_AST" in df.columns else ("AST" if has_ast else None)

        pts_allowed: Dict[str, float] = {}
        reb_allowed: Dict[str, float] = {}
        ast_allowed: Dict[str, float] = {}
        for _, row in df.iterrows():
            abbr = str(row[abbrev_col]).strip().upper()
            if not abbr:
                continue
            try:
                pts_allowed[abbr] = float(row[pts_col])
                if reb_col:
                    reb_allowed[abbr] = float(row[reb_col])
                if ast_col:
                    ast_allowed[abbr] = float(row[ast_col])
            except (TypeError, ValueError, KeyError):
                continue
        if not pts_allowed:
            _log_matchup_unavailable_once()
            return

        metrics_base: Dict[str, float] = {
            "pts_allowed_per_game": sum(pts_allowed.values()) / len(pts_allowed),
        }
        if reb_allowed:
            metrics_base["reb_allowed_per_game"] = sum(reb_allowed.values()) / len(reb_allowed)
        if ast_allowed:
            metrics_base["ast_allowed_per_game"] = sum(ast_allowed.values()) / len(ast_allowed)

        now = datetime.now(timezone.utc)
        cache = _load_cache()
        cache["league_avg"] = {"cached_at": now.isoformat(), "metrics": metrics_base}
        cache["teams"] = {}
        for abbr in pts_allowed:
            m: Dict[str, float] = {"pts_allowed_per_game": pts_allowed[abbr]}
            if abbr in reb_allowed:
                m["reb_allowed_per_game"] = reb_allowed[abbr]
            if abbr in ast_allowed:
                m["ast_allowed_per_game"] = ast_allowed[abbr]
            cache["teams"][abbr] = {"cached_at": now.isoformat(), "metrics": m}
        _save_cache(cache)
    except ImportError:
        _log_matchup_unavailable_once()
    except Exception:
        _log_matchup_unavailable_once()


def get_team_metrics(team_abbrev: str) -> Dict[str, float]:
    """
    Return team defensive metrics: pts_allowed_per_game, reb_allowed_per_game, ast_allowed_per_game.
    Only includes metrics actually fetched from API (no invented defaults for reb/ast).
    """
    if not team_abbrev or not team_abbrev.strip():
        return {"pts_allowed_per_game": DEFAULT_LEAGUE_PTS_ALLOWED}

    _ensure_team_metrics_cache()
    key = normalize_team_to_abbrev(team_abbrev)
    cache = _load_cache()
    teams = cache.get("teams", {})
    entry = teams.get(key)
    if entry is not None and not _is_expired(entry):
        m = {"pts_allowed_per_game": DEFAULT_LEAGUE_PTS_ALLOWED}
        m.update({k: v for k, v in entry.get("metrics", {}).items() if k in ("pts_allowed_per_game", "reb_allowed_per_game", "ast_allowed_per_game")})
        return m
    return {"pts_allowed_per_game": DEFAULT_LEAGUE_PTS_ALLOWED}


def get_league_avg_metrics() -> Dict[str, float]:
    """Return league-average defensive metrics. Only includes metrics actually fetched from API."""
    _ensure_team_metrics_cache()
    cache = _load_cache()
    entry = cache.get("league_avg")
    if entry is not None and not _is_expired(entry):
        m = entry.get("metrics", {})
        return {k: v for k, v in m.items() if k in ("pts_allowed_per_game", "reb_allowed_per_game", "ast_allowed_per_game")}
    return {"pts_allowed_per_game": DEFAULT_LEAGUE_PTS_ALLOWED}


def get_opponent_points_factor(team: str) -> float:
    """
    Multiplier around 1.0 for scoring vs this opponent.
    factor = league_avg_pts_allowed / opponent_pts_allowed.
    > 1 = easier matchup (opponent allows more); < 1 = tougher. Returns 1.0 if unknown.
    """
    if not team or not team.strip():
        return 1.0
    _ensure_team_metrics_cache()
    league_avg = get_league_avg_metrics().get("pts_allowed_per_game") or DEFAULT_LEAGUE_PTS_ALLOWED
    opponent_metrics = get_team_metrics(team)
    opponent_pts = opponent_metrics.get("pts_allowed_per_game") or league_avg
    if opponent_pts <= 0:
        return 1.0
    return league_avg / opponent_pts


def get_player_team(nba_player_id: int) -> Optional[str]:
    """Return current team abbreviation for the given NBA player id, or None."""
    try:
        from nba_api.stats.endpoints import commonplayerinfo

        info = commonplayerinfo.CommonPlayerInfo(player_id=nba_player_id)
        df = info.get_data_frames()[0]
        if df is not None and not df.empty:
            # COMMON_PLAYER_INFO has TEAM_ABBREVIATION or similar
            for col in ("TEAM_ABBREVIATION", "TEAM_ID", "ABBREVIATION"):
                if col in df.columns:
                    val = df[col].iloc[0]
                    if val and str(val).strip():
                        return str(val).strip()
        return None
    except Exception as e:
        print(f"DEBUG: get_player_team failed for {nba_player_id}: {e}")
        return None
