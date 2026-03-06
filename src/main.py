"""
Canonical ranking logic for PrimeProp. run_ranking_for_snapshot and helpers live here
so that importing src.main does not execute root main.py (no load_dotenv or entrypoint).
"""
import os
import asyncio
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.models import MarketSnapshot, Player
from src.optimizer import load_calibration_params, rank_props_by_edge
from src.projection import (
    ProjectionResult,
    get_projection_result as compute_projection_result,
    blend_short_long_result,
    ensemble_projection,
    winsorize,
    adjust_for_matchup,
    apply_context_adjustments,
    blend_lastN_with_season,
    MATCHUP_METRIC_PTS,
    MATCHUP_METRIC_REB,
    MATCHUP_METRIC_AST,
    compute_stdev,
)
from src.stats import (
    fetch_last_n_games,
    get_stat_series,
    get_stat_and_minutes_series,
    get_rest_days,
    fetch_season_to_date_avg,
    resolve_nba_player_id,
)
from src.matchup import get_team_metrics, get_league_avg_metrics, get_player_team, normalize_team_to_abbrev


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _enrich_ranked_with_line_movement(
    ranked: List[Any],
    open_lines: Optional[Dict[Tuple[str, str], float]],
) -> List[Any]:
    """Set open_line, current_line, delta_line on each PropEdge from open_lines. Returns new list."""
    if not open_lines:
        return ranked
    out = []
    for e in ranked:
        open_ln = open_lines.get((e.player_id, e.stat_type), e.market_line)
        cur_ln = e.market_line
        delta = cur_ln - open_ln
        out.append(e.model_copy(update={"open_line": open_ln, "current_line": cur_ln, "delta_line": delta}))
    return out


async def run_ranking_for_snapshot(
    snapshot: MarketSnapshot,
    players_by_id: Dict[str, Player],
    *,
    ingest_counters: Optional[Dict[str, Any]] = None,
    open_lines: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[List[Any], Dict[str, str], Dict[str, Optional[int]], Dict[Tuple[str, str], Tuple[float, Optional[ProjectionResult], List[float]]]]:
    """
    Build projections (with variance inflation via get_projection_result), apply minutes + context + matchup,
    rank by EV, enrich line movement. Returns (ranked, id_to_canonical, resolved_nba_ids, projected) for main.
    """
    if ingest_counters is None:
        ingest_counters = {}

    # Projection config from env
    projection_n_games = _env_int("PROJECTION_N_GAMES", 10)
    min_games_for_projection = _env_int("MIN_GAMES_FOR_PROJECTION", 5)
    winsorize_pct = _env_float("WINSORIZE_PCT", 0.0)
    stats_max_concurrency = _env_int("STATS_MAX_CONCURRENCY", 8)
    projection_method = (os.getenv("PROJECTION_METHOD", "weighted_average") or "weighted_average").strip().lower()
    if projection_method not in ("weighted_average", "simple_average", "exponential", "blend_short_long"):
        projection_method = "weighted_average"

    projection_strategy = (os.getenv("PROJECTION_STRATEGY", "single") or "single").strip().lower()
    blend_enabled = projection_strategy == "blend"
    short_n = _env_int("SHORT_N", 5)
    long_n = _env_int("LONG_N", 15)
    regression_alpha = _env_float("REGRESSION_ALPHA", 0.65)
    ensemble_enabled = projection_strategy == "ensemble"
    ensemble_weights_str = (os.getenv("ENSEMBLE_WEIGHTS", "weighted_average:0.6,simple_average:0.4") or "").strip()
    season_blend_enabled = projection_strategy == "season_blend"
    season_blend_alpha = _env_float("SEASON_BLEND_ALPHA", 0.7)
    matchup_enabled = _env_bool("MATCHUP_ENABLED", True)
    matchup_strength = _env_float("MATCHUP_STRENGTH", 0.5)
    min_avg_minutes = _env_float("MIN_AVG_MINUTES", 18.0)
    max_minutes_stdev = _env_float("MAX_MINUTES_STDEV", 6.0)
    trend_z_threshold = _env_float("TREND_Z_THRESHOLD", 1.0)
    trend_stdev_multiplier = _env_float("TREND_STDEV_MULTIPLIER", 1.3)
    ev_threshold: Optional[float] = None
    try:
        ev_str = os.getenv("EV_THRESHOLD", "").strip()
        if ev_str:
            ev_threshold = float(ev_str)
    except (ValueError, TypeError):
        pass

    # Unique (player_id, stat_type) and id -> canonical name
    unique_keys = set((line.player_id, line.stat_type) for line in snapshot.lines)
    id_to_canonical: Dict[str, str] = {}
    player_to_stat_types: Dict[str, List[str]] = {}
    for pid, st in unique_keys:
        if pid not in id_to_canonical:
            p = players_by_id.get(pid)
            id_to_canonical[pid] = p.canonical_name if p else pid
        player_to_stat_types.setdefault(pid, []).append(st)

    skipped_small_sample = 0
    failed_fetch = 0
    trend_flagged_count = 0
    resolved_nba_ids: Dict[str, Optional[int]] = {}
    for pid in player_to_stat_types:
        canonical = id_to_canonical.get(pid, pid)
        nba_id = resolve_nba_player_id(canonical)
        resolved_nba_ids[pid] = nba_id

    if hasattr(snapshot, "timestamp") and snapshot.timestamp:
        ts = snapshot.timestamp
        target_game_date = ts.date() if isinstance(ts, datetime) else date.today()
    else:
        target_game_date = date.today()

    # Per-player game context: each player gets (home_team, away_team) from the line(s) they appear in (multi-game slate).
    player_id_to_game: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for line in snapshot.lines:
        if line.home_team or line.away_team:
            player_id_to_game[line.player_id] = (line.home_team, line.away_team)

    league_avg_metrics = get_league_avg_metrics()
    matchup_metric_map = {"Points": MATCHUP_METRIC_PTS, "Rebounds": MATCHUP_METRIC_REB, "Assists": MATCHUP_METRIC_AST}
    n_games_fetch = long_n if (blend_enabled or ensemble_enabled) else projection_n_games
    n_games_fetch = max(n_games_fetch, projection_n_games)

    ensemble_weights_list: List[Tuple[str, float]] = []
    if ensemble_enabled and ensemble_weights_str:
        for part in ensemble_weights_str.split(","):
            part = part.strip()
            if ":" in part:
                name, w = part.split(":", 1)
                try:
                    ensemble_weights_list.append((name.strip(), float(w.strip())))
                except (ValueError, TypeError):
                    pass
    if not ensemble_weights_list:
        ensemble_weights_list = [("weighted_average", 0.6), ("simple_average", 0.4)]

    projected: Dict[Tuple[str, str], Tuple[float, Optional[ProjectionResult], List[float]]] = {}
    semaphore = asyncio.Semaphore(stats_max_concurrency)

    async def process_player(player_id: str, stat_types: List[str]) -> None:
        nonlocal skipped_small_sample, failed_fetch, trend_flagged_count
        nba_id = resolved_nba_ids.get(player_id)
        if nba_id is None:
            failed_fetch += 1
            return
        async with semaphore:
            try:
                games = await fetch_last_n_games(nba_id, n_games_fetch)
            except Exception:
                failed_fetch += 1
                return
        if not games:
            failed_fetch += 1
            return

        # Pre-filter: most recent game DNP/0 minutes -> skip player
        last_game = games[-1]
        last_min = last_game.get("MIN")
        if not isinstance(last_min, (int, float)) or last_min <= 0:
            ingest_counters["skipped_recent_dnp"] = ingest_counters.get("skipped_recent_dnp", 0) + 1
            return

        # Minutes stability filter (MIN_AVG_MINUTES, MAX_MINUTES_STDEV)
        mins = [g["MIN"] for g in games if "MIN" in g]
        mins = [m for m in mins if m is not None and isinstance(m, (int, float))]
        avg_minutes = None
        minutes_stdev = None
        if mins:
            n_min = len(mins)
            avg_minutes = sum(mins) / n_min
            if n_min >= 2:
                minutes_stdev = compute_stdev(mins)
            else:
                minutes_stdev = 0.0
        if avg_minutes is not None and avg_minutes < min_avg_minutes:
            ingest_counters["skipped_low_minutes"] = ingest_counters.get("skipped_low_minutes", 0) + 1
            return
        if minutes_stdev is not None and minutes_stdev > max_minutes_stdev:
            ingest_counters["skipped_unstable_minutes"] = ingest_counters.get("skipped_unstable_minutes", 0) + 1
            return

        player_team_abbrev: Optional[str] = None
        game_home_away = player_id_to_game.get(player_id, (None, None))
        home_team, away_team = game_home_away
        if matchup_enabled:
            player_team_abbrev = get_player_team(nba_id)
        opponent_metrics_pts = league_avg_metrics
        opponent_metrics_reb = league_avg_metrics
        opponent_metrics_ast = league_avg_metrics
        if matchup_enabled and home_team and away_team and player_team_abbrev:
            home_abbrev = normalize_team_to_abbrev(home_team)
            away_abbrev = normalize_team_to_abbrev(away_team)
            opponent_abbrev = away_abbrev if player_team_abbrev == home_abbrev else home_abbrev
            opponent_metrics_pts = get_team_metrics(opponent_abbrev)
            opponent_metrics_reb = get_team_metrics(opponent_abbrev)
            opponent_metrics_ast = get_team_metrics(opponent_abbrev)

        rest_days = get_rest_days(games, target_game_date)
        is_home = None
        if home_team and player_team_abbrev:
            home_abbrev = normalize_team_to_abbrev(home_team)
            is_home = player_team_abbrev == home_abbrev

        for stat_type in stat_types:
            values, mins = get_stat_and_minutes_series(games, stat_type)
            if not values:
                continue
            if winsorize_pct > 0 and winsorize_pct < 0.5:
                values, _ = winsorize(values, winsorize_pct)
                # mins stay aligned; winsorize does not change length

            if len(values) < min_games_for_projection:
                skipped_small_sample += 1
                continue

            result: Optional[ProjectionResult] = None
            mean_val: float = 0.0

            if blend_enabled:
                short_vals = values[-short_n:] if len(values) >= short_n else values
                long_vals = values[-long_n:] if len(values) >= long_n else values
                res, _ = blend_short_long_result(
                    short_vals, long_vals, regression_alpha, projection_method,
                    stat_type=stat_type, min_games=min_games_for_projection,
                )
                if res is not None:
                    result, mean_val = res, res.mean
            elif ensemble_enabled:
                values_short = values[-short_n:] if len(values) >= short_n else values
                values_long = values[-long_n:] if len(values) >= long_n else values
                res, _ = ensemble_projection(
                    ensemble_weights_list, values_short, values_long, regression_alpha,
                    stat_type=stat_type, min_games=min_games_for_projection,
                )
                if res is not None:
                    result, mean_val = res, res.mean
            elif season_blend_enabled:
                base_result = compute_projection_result(
                    player_id, stat_type, values,
                    historical_minutes=mins,
                    n_games=projection_n_games, method=projection_method, min_games=min_games_for_projection,
                )
                if base_result is not None:
                    try:
                        season_avg = await fetch_season_to_date_avg(nba_id, stat_type)
                        if season_avg is not None:
                            result = blend_lastN_with_season(base_result, season_avg, season_blend_alpha)
                            mean_val = result.mean
                        else:
                            result, mean_val = base_result, base_result.mean
                    except Exception:
                        result, mean_val = base_result, base_result.mean
            else:
                result = compute_projection_result(
                    player_id, stat_type, values,
                    historical_minutes=mins,
                    n_games=projection_n_games, method=projection_method, min_games=min_games_for_projection,
                )
                mean_val = result.mean if result else 0.0

            if result is None:
                continue

            # Context adjustments (is_home, rest_days)
            if is_home is not None and rest_days is not None:
                mean_val = apply_context_adjustments(mean_val, stat_type, is_home, rest_days)
                result = ProjectionResult(mean=mean_val, stdev=result.stdev, n=result.n, confidence=result.confidence)

            # Matchup
            if matchup_enabled:
                metric_key = matchup_metric_map.get(stat_type, MATCHUP_METRIC_PTS)
                opp = opponent_metrics_pts if stat_type == "Points" else (opponent_metrics_reb if stat_type == "Rebounds" else opponent_metrics_ast)
                mean_val = adjust_for_matchup(mean_val, opp, league_avg_metrics, matchup_strength, metric_key)
                result = ProjectionResult(mean=mean_val, stdev=result.stdev, n=result.n, confidence=result.confidence)

            # Trend-based instability: short (3) vs medium (10) window; flag if |short_mean - medium_mean| > z * stdev_medium
            if len(values) >= 3:
                short_vals = values[-3:]
                medium_vals = values[-10:] if len(values) >= 10 else values
                short_mean = sum(short_vals) / len(short_vals)
                medium_mean = sum(medium_vals) / len(medium_vals)
                trend_delta = short_mean - medium_mean
                stdev_medium = compute_stdev(medium_vals)
                if stdev_medium > 1e-9 and abs(trend_delta) > trend_z_threshold * stdev_medium:
                    trend_flagged_count += 1
                    result = ProjectionResult(
                        mean=result.mean,
                        stdev=result.stdev * trend_stdev_multiplier,
                        n=result.n,
                        confidence="high_variance",
                    )

            projected[(player_id, stat_type)] = (mean_val, result, values)

    await asyncio.gather(*(process_player(pid, st_list) for pid, st_list in player_to_stat_types.items()))

    summary_parts = [f"failed_fetch={failed_fetch}", f"skipped_small_sample={skipped_small_sample}", f"trend_flagged_count={trend_flagged_count}"]
    if ingest_counters.get("skipped_recent_dnp"):
        summary_parts.append(f"skipped_recent_dnp={ingest_counters['skipped_recent_dnp']}")
    if ingest_counters.get("skipped_low_minutes"):
        summary_parts.append(f"skipped_low_minutes={ingest_counters['skipped_low_minutes']}")
    if ingest_counters.get("skipped_unstable_minutes"):
        summary_parts.append(f"skipped_unstable_minutes={ingest_counters['skipped_unstable_minutes']}")
    print("Projection summary: " + ", ".join(summary_parts))

    def get_projection(pid: str, st: str) -> Optional[float]:
        entry = projected.get((pid, st), (None, None, None))
        return entry[0]

    def lookup_projection_result(pid: str, st: str) -> Optional[ProjectionResult]:
        entry = projected.get((pid, st), (None, None, None))
        return entry[1]

    def get_last_n_values(pid: str, st: str) -> Optional[List[float]]:
        entry = projected.get((pid, st), (None, None, None))
        return entry[2] if len(entry) > 2 else None

    calibration_params = load_calibration_params()
    if calibration_params is not None:
        a, b = calibration_params
        print("Calibration: a={:.4f}, b={:.4f}".format(a, b))
    else:
        print("Calibration: none")

    ranked = rank_props_by_edge(
        snapshot,
        get_projection,
        get_projection_result=lookup_projection_result,
        get_last_n_values=get_last_n_values,
        calibration_params=calibration_params,
        ev_threshold=ev_threshold,
    )
    ranked = _enrich_ranked_with_line_movement(ranked, open_lines)

    return (ranked, id_to_canonical, resolved_nba_ids, projected)
