"""
PrimeProp main entrypoint: fetch PRE-GAME NBA odds from The Odds API,
run the optimizer, and print/alert the top +EV props.
"""
import json
import os
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # MUST BE LOADED BEFORE SRC MODULES

from src.database import DatabaseManager
from src.ingest import (
    OddsApiIngestor,
    fetch_multi_source_snapshot,
    aggregate_snapshot_by_best_odds,
    AGGREGATE_STAT_TYPES,
)
from src.optimizer import rank_props_by_edge, load_calibration, DEFAULT_CALIBRATION_PATH
from src.projection import (
    StatType,
    get_projection_result,
    blend_short_long_result,
    ensemble_projection,
    ProjectionResult,
    winsorize,
)
from src.alerting import alert_high_value_props
from src.stats import fetch_last_n_games, get_stat_series, resolve_nba_player_id
from src.matchup import get_team_metrics, get_league_avg_metrics, get_player_team, normalize_team_to_abbrev
from src.projection import (
    adjust_for_matchup,
    MATCHUP_METRIC_PTS,
    MATCHUP_METRIC_REB,
    MATCHUP_METRIC_AST,
)


async def get_upcoming_event_ids(api_key: str) -> list[str]:
    """Fetches games and returns IDs only for games in today's betting day that have NOT started yet."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    upcoming_ids = []
    now = datetime.now(timezone.utc)
    current_betting_date = (now - timedelta(hours=6)).date()

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params={"apiKey": api_key}) as resp:
            games = await resp.json()

            for game in games:
                # API returns time like "2024-10-22T23:30:00Z", we make it Python-friendly
                commence_str = game["commence_time"].replace("Z", "+00:00")
                commence_time = datetime.fromisoformat(commence_str)

                # Implement a 1:00 AM EST betting-day rollover (UTC-5 ≈ 6-hour shift).
                game_betting_date = (commence_time - timedelta(hours=6)).date()

                # Only grab games where the tip-off time is in the future AND in today's betting day.
                if game_betting_date == current_betting_date and commence_time > now:
                    upcoming_ids.append(game["id"])

    return upcoming_ids


async def main() -> None:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not set in environment. Add it to .env")

    print("Checking for upcoming pre-game matchups...")
    upcoming_event_ids = await get_upcoming_event_ids(api_key)
    if not upcoming_event_ids:
        print("No more upcoming games for today. All matchups have tipped off. Check back tomorrow morning for the new slate!")
        return

    print(f"Found {len(upcoming_event_ids)} upcoming games. Fetching pre-game player props...")
    # We dynamically create an Ingestor for EVERY upcoming game
    ingestors = []
    for event_id in upcoming_event_ids:
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
        ingestors.append(
            OddsApiIngestor(
                url,
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "bookmakers": "fanduel,draftkings,betmgm",
                    "markets": "player_points,player_rebounds,player_assists",
                    "oddsFormat": "american",
                },
            )
        )

    # Pass ALL the ingestors into your beautifully built concurrent snapshot fetcher
    snapshot, players_by_id, ingest_counters = await fetch_multi_source_snapshot(
        snapshot_id="pre-game-1",
        game_id="upcoming_nba_slate",
        players=[],
        providers=ingestors,
    )
    print(f"DEBUG: Total props ingested: {len(snapshot.lines)}")

    # Aggregate by (player_id, stat_type, line); pick best odds per side across books
    snapshot = aggregate_snapshot_by_best_odds(snapshot, stat_types=AGGREGATE_STAT_TYPES)
    print(f"DEBUG: Props after aggregation (PTS/REB/AST): {len(snapshot.lines)}")

    # Projection config from env
    try:
        projection_n_games = int(os.getenv("PROJECTION_N_GAMES", "10").strip())
    except ValueError:
        projection_n_games = 10
    if projection_n_games < 1 or projection_n_games > 100:
        projection_n_games = 10
    try:
        min_games_for_projection = int(os.getenv("MIN_GAMES_FOR_PROJECTION", "5").strip())
    except ValueError:
        min_games_for_projection = 5
    if min_games_for_projection < 1:
        min_games_for_projection = 5
    try:
        winsorize_pct = float(os.getenv("WINSORIZE_PCT", "0.05").strip())
    except ValueError:
        winsorize_pct = 0.0
    if winsorize_pct <= 0 or winsorize_pct >= 0.5:
        winsorize_pct = 0.0
    try:
        stats_max_concurrency = int(os.getenv("STATS_MAX_CONCURRENCY", "5").strip())
    except ValueError:
        stats_max_concurrency = 5
    if stats_max_concurrency < 1:
        stats_max_concurrency = 5
    projection_method = (os.getenv("PROJECTION_METHOD", "weighted_average") or "weighted_average").strip()

    # Regression-to-mean blending (short/long): check BLEND_REGRESSION_TO_MEAN first, fallback to BLEND_REGESSION_TO_MEAN
    blend_val = os.getenv("BLEND_REGRESSION_TO_MEAN") or os.getenv("BLEND_REGESSION_TO_MEAN") or ""
    blend_enabled = blend_val.strip().lower() in ("1", "true", "yes")
    try:
        short_n = int(os.getenv("SHORT_N", "5").strip())
    except ValueError:
        short_n = 5
    try:
        long_n = int(os.getenv("LONG_N", "20").strip())
    except ValueError:
        long_n = 20
    try:
        regression_alpha = float(os.getenv("REGRESSION_ALPHA", "0.65").strip())
    except ValueError:
        regression_alpha = 0.65
    if regression_alpha < 0 or regression_alpha > 1:
        regression_alpha = 0.65

    # Projection strategy: single vs ensemble
    projection_strategy = (os.getenv("PROJECTION_STRATEGY", "single") or "single").strip().lower()
    ensemble_enabled = projection_strategy == "ensemble"

    # Parse ENSEMBLE_WEIGHTS: JSON {"wa": 0.5, ...} or comma "weighted_average:0.5,simple_average:0.2,blend_short_long:0.3"
    ensemble_weights_raw = (os.getenv("ENSEMBLE_WEIGHTS") or "weighted_average:0.5,simple_average:0.2,blend_short_long:0.3").strip()
    ensemble_weights_list: list[tuple[str, float]] = []
    if ensemble_enabled:
        if ensemble_weights_raw.startswith("{"):
            try:
                d = json.loads(ensemble_weights_raw)
                ensemble_weights_list = [(k, float(v)) for k, v in d.items()]
            except (json.JSONDecodeError, ValueError):
                ensemble_weights_list = [("weighted_average", 0.5), ("simple_average", 0.2), ("blend_short_long", 0.3)]
        else:
            for part in ensemble_weights_raw.split(","):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    try:
                        ensemble_weights_list.append((k.strip(), float(v.strip())))
                    except ValueError:
                        pass
        if not ensemble_weights_list:
            ensemble_weights_list = [("weighted_average", 0.5), ("simple_average", 0.2), ("blend_short_long", 0.3)]

    # Matchup (opponent) adjustment: ENABLE_MATCHUP_ADJUSTMENT default false, MATCHUP_STRENGTH default 0.4
    matchup_enabled = os.getenv("ENABLE_MATCHUP_ADJUSTMENT", "").strip().lower() in ("1", "true", "yes")
    try:
        matchup_strength = float(os.getenv("MATCHUP_STRENGTH", "0.4").strip())
    except ValueError:
        matchup_strength = 0.4
    matchup_strength = max(0.0, min(1.0, matchup_strength))
    if matchup_enabled:
        print(f"Projection config: matchup adjustment enabled, strength={matchup_strength}")

    if ensemble_enabled:
        print(
            f"Projection config: strategy=ensemble, short_n={short_n}, long_n={long_n}, "
            f"blend_alpha={regression_alpha}, weights={ensemble_weights_list}, min_games={min_games_for_projection}"
        )
    elif blend_enabled:
        print(
            f"Projection config: blend_short_long enabled, short_n={short_n}, long_n={long_n}, "
            f"alpha={regression_alpha}, base_method={projection_method!r}, min_games={min_games_for_projection}"
        )
    else:
        print(
            f"Projection config: n_games={projection_n_games}, method={projection_method!r}, "
            f"min_games={min_games_for_projection}, stats_max_concurrency={stats_max_concurrency}"
        )

    # Build projections: group by player, fetch gamelog once per player, then series per stat type.
    print("Fetching historical stats and building projections...")
    projections: dict[tuple[str, StatType], ProjectionResult] = {}
    unique_keys = {(line.player_id, line.stat_type) for line in snapshot.lines}
    id_to_canonical = {pid: p.canonical_name for pid, p in players_by_id.items()}

    # Group by player_id -> set of stat_types
    player_to_stat_types: dict[str, set[StatType]] = {}
    for (player_id, stat_type) in unique_keys:
        player_to_stat_types.setdefault(player_id, set()).add(stat_type)

    total_keys = len(unique_keys)
    skipped_small_sample = 0
    failed_fetch = 0
    resolved_nba_id = 0
    unresolved_player_skips = 0
    winsorized_count = 0
    confidence_counts: dict[str, int] = {"insufficient": 0, "low": 0, "high_variance": 0, "ok": 0}
    strategy_counts: dict[str, int] = {}
    resolved_nba_ids: dict[str, int] = {}  # player_id -> nba_player_id for DB logging
    matchup_metric_warned: set[str] = set()  # stat_types we've already printed "metric not present" for

    print(f"DEBUG: Unique (player, stat_type) pairs found: {total_keys}")

    # Build (player_id, stat_type) -> (home_team, away_team) from lines
    game_context: dict[tuple[str, StatType], tuple[str | None, str | None]] = {}
    for line in snapshot.lines:
        key = (line.player_id, line.stat_type)
        if key not in game_context and (getattr(line, "home_team", None) or getattr(line, "away_team", None)):
            game_context[key] = (getattr(line, "home_team", None), getattr(line, "away_team", None))

    league_avg_metrics = get_league_avg_metrics() if matchup_enabled else {}
    n_games_fetch = long_n if (blend_enabled or ensemble_enabled) else projection_n_games
    semaphore = asyncio.Semaphore(stats_max_concurrency)
    counters_lock = asyncio.Lock()

    async def process_player(player_id: str, stat_types: set[StatType]) -> None:
        nonlocal resolved_nba_id, unresolved_player_skips, failed_fetch, skipped_small_sample, winsorized_count, resolved_nba_ids, matchup_metric_warned
        player = players_by_id.get(player_id)
        if not player:
            return
        nba_id = player.nba_player_id or await asyncio.to_thread(
            resolve_nba_player_id, player.canonical_name
        )
        if nba_id is None:
            async with counters_lock:
                unresolved_player_skips += len(stat_types)
            return
        async with counters_lock:
            resolved_nba_id += 1
            resolved_nba_ids[player_id] = nba_id
        display_name = player.canonical_name
        async with semaphore:
            games = await fetch_last_n_games(nba_id, n_games_fetch)
        if not games:
            async with counters_lock:
                failed_fetch += len(stat_types)
            return
        for stat_type in stat_types:
            values = get_stat_series(games, stat_type)
            if not values:
                async with counters_lock:
                    failed_fetch += 1
                continue
            if len(values) < min_games_for_projection:
                async with counters_lock:
                    skipped_small_sample += 1
                continue
            if winsorize_pct > 0:
                values, wc = winsorize(values, winsorize_pct)
                async with counters_lock:
                    winsorized_count += wc
            if ensemble_enabled:
                short_values = values[-short_n:] if len(values) >= short_n else values
                long_values = values[-long_n:] if len(values) >= long_n else values
                result, strategy_means = ensemble_projection(
                    ensemble_weights_list,
                    short_values,
                    long_values,
                    blend_alpha=regression_alpha,
                    min_games=min_games_for_projection,
                )
                strategy = "ensemble"
                if result is not None and strategy_means:
                    parts = [f"{k}={v:.2f}" for k, v in sorted(strategy_means.items())]
                    print(
                        f"  Ensemble {display_name} {stat_type}: "
                        f"{', '.join(parts)} -> mean={result.mean:.2f}"
                    )
            elif blend_enabled:
                short_values = values[-short_n:] if len(values) >= short_n else values
                long_values = values[-long_n:] if len(values) >= long_n else values
                result, strategy = blend_short_long_result(
                    short_values, long_values, regression_alpha, projection_method,
                    min_games=min_games_for_projection,
                )
            else:
                result = get_projection_result(
                    player_id=player_id,
                    stat_type=stat_type,
                    historical_values=values,
                    n_games=projection_n_games,
                    method=projection_method,
                    min_games=min_games_for_projection,
                )
                strategy = projection_method
            if result is None:
                async with counters_lock:
                    failed_fetch += 1
                continue
            # Matchup adjustment: PTS/REB/AST when matchup.py has the metric (from LeagueDashOpponentTeamStats).
            if matchup_enabled and league_avg_metrics:
                metric_key = None
                if stat_type == "Points" and MATCHUP_METRIC_PTS in league_avg_metrics:
                    metric_key = MATCHUP_METRIC_PTS
                elif stat_type == "Rebounds" and MATCHUP_METRIC_REB in league_avg_metrics:
                    metric_key = MATCHUP_METRIC_REB
                elif stat_type == "Assists" and MATCHUP_METRIC_AST in league_avg_metrics:
                    metric_key = MATCHUP_METRIC_AST
                if metric_key is None and stat_type in ("Points", "Rebounds", "Assists"):
                    needed = MATCHUP_METRIC_PTS if stat_type == "Points" else (MATCHUP_METRIC_REB if stat_type == "Rebounds" else MATCHUP_METRIC_AST)
                    async with counters_lock:
                        if stat_type not in matchup_metric_warned:
                            print(f"DEBUG: matchup adjustment skipped for {stat_type}: {needed} not in league_avg_metrics (API may not return it).")
                            matchup_metric_warned.add(stat_type)
                if metric_key:
                    home_team, away_team = game_context.get((player_id, stat_type), (None, None))
                    opponent_team = None
                    if nba_id and home_team and away_team:
                        player_team_abbrev = get_player_team(nba_id)
                        if player_team_abbrev:
                            ht_abbrev = normalize_team_to_abbrev(home_team or "")
                            if player_team_abbrev.upper() == ht_abbrev:
                                opponent_team = away_team
                            else:
                                opponent_team = home_team
                    opponent_abbrev = normalize_team_to_abbrev(opponent_team) if opponent_team else None
                    if opponent_abbrev:
                        opponent_metrics = get_team_metrics(opponent_abbrev)
                        if metric_key in opponent_metrics:
                            baseline_mean = result.mean
                            adjusted_mean = adjust_for_matchup(
                                baseline_mean,
                                opponent_metrics,
                                league_avg_metrics,
                                matchup_strength,
                                metric_key=metric_key,
                            )
                            if abs(adjusted_mean - baseline_mean) > 0.01:
                                print(
                                    f"  Matchup {display_name} {stat_type}: baseline={baseline_mean:.2f} "
                                    f"opponent={opponent_abbrev} -> adjusted={adjusted_mean:.2f}"
                                )
                            result = ProjectionResult(mean=adjusted_mean, stdev=result.stdev, n=result.n, confidence=result.confidence)
            projections[(player_id, stat_type)] = result
            async with counters_lock:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                conf = result.confidence
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    await asyncio.gather(
        *(process_player(pid, st) for pid, st in player_to_stat_types.items())
    )

    projected = len(projections)
    print(
        f"Projection summary: total_keys={total_keys}, projected={projected}, "
        f"skipped_small_sample={skipped_small_sample}, failed_fetch={failed_fetch}, winsorized_count={winsorized_count}."
    )
    print(
        f"Player identity: resolved_nba_id={resolved_nba_id}, unresolved_player_skips={unresolved_player_skips}, "
        f"duplicate_player_alias_added={ingest_counters.get('duplicate_player_alias_added', 0)}"
    )
    print(
        f"Projection strategy: {', '.join(f'{k}={v}' for k, v in sorted(strategy_counts.items()))}."
    )
    print(
        f"Projection confidence: insufficient={confidence_counts['insufficient']}, low={confidence_counts['low']}, "
        f"high_variance={confidence_counts['high_variance']}, ok={confidence_counts['ok']}."
    )

    def projection_provider(player_id: str, stat_type: StatType) -> float | None:
        r = projections.get((player_id, stat_type))
        return r.mean if r is not None else None

    def get_projection_result(player_id: str, stat_type: StatType) -> ProjectionResult | None:
        return projections.get((player_id, stat_type))

    # Optional probability calibration: load from calibration_params.json (project root); apply only for EV; store raw p_over_model
    calibration_params = load_calibration(DEFAULT_CALIBRATION_PATH)
    if calibration_params is not None:
        print(f"Calibration: using params a={calibration_params[0]:.4f}, b={calibration_params[1]:.4f}")

    ev_threshold = float(os.getenv("EV_THRESHOLD", "0.02").strip())

    ranked = rank_props_by_edge(
        snapshot,
        projection_provider,
        get_projection_result=get_projection_result,
        calibration_params=calibration_params,
        ev_threshold=ev_threshold,
    )

    print("\nTop 5 PRE-GAME +EV bets (ranked by best_ev):")
    for i, edge in enumerate(ranked[:5], 1):
        name = id_to_canonical.get(edge.player_id, edge.player_id)
        proj_s = f"{edge.projected:.1f}"
        if edge.projected_stdev is not None:
            proj_s += f"±{edge.projected_stdev:.1f}"
        ev_s = f"EV={edge.best_ev * 100:.2f}%" if edge.best_ev is not None else f"edge={edge.edge * 100:.2f}%"
        book_s = edge.recommended_provider or edge.provider
        odds_s = f" @ {edge.recommended_odds:+.0f}" if edge.recommended_odds is not None else ""
        print(
            f"  {i}. {name} | {edge.stat_type} {edge.recommended_side} {edge.market_line} | "
            f"Proj: {proj_s} | {ev_s} | {book_s}{odds_s}"
        )

    # Fire off alerts (filter by best_ev when available)
    high_value_alerts = alert_high_value_props(
        ranked, min_edge=0.05, min_ev=0.05, player_names=id_to_canonical
    )
    print(f"\nSuccessfully fired alerts for {len(high_value_alerts)} high-value pre-game props!")

    # Persist high-value picks to database (only if we have any)
    if high_value_alerts:
        db = DatabaseManager()
        for edge in high_value_alerts:
            db.log_pick(
                player_name=id_to_canonical.get(edge.player_id, edge.player_id),
                stat_type=edge.stat_type,
                market_line=edge.market_line,
                projected=edge.projected,
                edge=edge.edge,
                recommended_side=edge.recommended_side,
                p_over_model=edge.p_over_model,
                stdev=edge.projected_stdev,
                odds=edge.recommended_odds,
                over_odds=edge.over_odds,
                under_odds=edge.under_odds,
                over_provider=edge.over_provider,
                under_provider=edge.under_provider,
                player_id=edge.player_id,
                nba_player_id=resolved_nba_ids.get(edge.player_id),
                game_start=None,
            )
        print(f"Saved {len(high_value_alerts)} picks to the database.")
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
