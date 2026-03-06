"""
PrimeProp main entrypoint: fetch PRE-GAME NBA odds from The Odds API,
run the optimizer, and print/alert the top +EV props.

Ranking logic lives in src.main; this module is a thin entrypoint that loads .env and calls run_ranking_for_snapshot.
"""
import os
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()  # MUST BE LOADED BEFORE SRC MODULES

from src.database import DatabaseManager
from src.ingest import (
    OddsApiIngestor,
    fetch_multi_source_snapshot,
    aggregate_snapshot_by_best_odds,
    AGGREGATE_STAT_TYPES,
)
from src.alerting import alert_high_value_props
from src.models import MarketSnapshot, Player
from src.injury_filter import (
    is_injury_filter_enabled,
    get_unavailable_player_ids,
)
from src.main import run_ranking_for_snapshot, _env_int, _env_float, _env_bool


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
                commence_str = game["commence_time"].replace("Z", "+00:00")
                commence_time = datetime.fromisoformat(commence_str)
                game_betting_date = (commence_time - timedelta(hours=6)).date()
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

    snapshot, players_by_id, ingest_counters = await fetch_multi_source_snapshot(
        snapshot_id="pre-game-1",
        game_id="upcoming_nba_slate",
        players=[],
        providers=ingestors,
    )
    print(f"DEBUG: Total props ingested: {len(snapshot.lines)}")

    snapshot = aggregate_snapshot_by_best_odds(snapshot, stat_types=AGGREGATE_STAT_TYPES)
    print(f"DEBUG: Props after aggregation (PTS/REB/AST): {len(snapshot.lines)}")

    # Injury/availability pre-filter: exclude props for OUT/DOUBTFUL/QUESTIONABLE (configurable) and override list
    if is_injury_filter_enabled():
        unavailable = get_unavailable_player_ids(players_by_id, api_key=api_key)
        n_before = len(snapshot.lines)
        filtered_lines = [line for line in snapshot.lines if line.player_id not in unavailable]
        snapshot = snapshot.model_copy(update={"lines": filtered_lines})
        skipped_injury_count = n_before - len(snapshot.lines)
        print("Injury filter: skipped_injury_count={}".format(skipped_injury_count))

    open_lines: Optional[Dict[Tuple[str, str], float]] = None
    _db = DatabaseManager()
    try:
        open_lines = _db.get_earliest_snapshot_lines_for_game(snapshot.game_id)
        if open_lines:
            print(f"DEBUG: Line movement: using {len(open_lines)} open lines from earliest snapshot today.")
        if os.getenv("PERSIST_SNAPSHOTS", "0").strip().lower() in ("1", "true", "yes"):
            pk = _db.save_snapshot(snapshot, players_by_id)
            print(f"DEBUG: Persisted snapshot id={pk} ({snapshot.snapshot_id}, {len(snapshot.lines)} lines).")
    finally:
        _db.close()

    ranked, id_to_canonical, resolved_nba_ids, projected = await run_ranking_for_snapshot(
        snapshot, players_by_id, ingest_counters=ingest_counters, open_lines=open_lines
    )

    if os.getenv("REQUIRE_LINE_MOVEMENT_WITH_US", "0").strip().lower() in ("1", "true", "yes"):
        delta_default = 0.0
        ranked = [
            e for e in ranked
            if (e.recommended_side == "Over" and (e.delta_line if e.delta_line is not None else delta_default) < 0)
            or (e.recommended_side == "Under" and (e.delta_line if e.delta_line is not None else delta_default) > 0)
        ]
        print(f"DEBUG: Filtered to {len(ranked)} props with line movement in our favor (REQUIRE_LINE_MOVEMENT_WITH_US=1).")

    # Confidence-based filter: exclude high_variance by default; allow only if INCLUDE_HIGH_VARIANCE=1 and best_ev >= HIGH_VARIANCE_MIN_EV
    include_high_variance = _env_int("INCLUDE_HIGH_VARIANCE", 0)
    high_variance_min_ev = _env_float("HIGH_VARIANCE_MIN_EV", 0.08)
    excluded_high_variance = 0
    included_high_variance_due_to_ev = 0

    def _confidence_for_edge(edge: Any) -> str:
        res = projected.get((edge.player_id, edge.stat_type), (None, None, None))[1]
        return getattr(res, "confidence", "ok") if res else "ok"

    filtered_ranked: List[Any] = []
    for e in ranked:
        conf = _confidence_for_edge(e)
        if conf != "high_variance":
            filtered_ranked.append(e)
            continue
        if include_high_variance == 0:
            excluded_high_variance += 1
            continue
        best_ev = getattr(e, "best_ev", None)
        if best_ev is not None and best_ev >= high_variance_min_ev:
            filtered_ranked.append(e)
            included_high_variance_due_to_ev += 1
        else:
            excluded_high_variance += 1
    ranked = filtered_ranked
    print(f"High-variance filter: excluded_high_variance={excluded_high_variance}, included_high_variance_due_to_ev={included_high_variance_due_to_ev}")

    # Top N (EV ranking): mean±stdev, EV, book/odds, confidence label
    top_n = 10
    print(f"\n--- Top {top_n} props by edge/EV ---")
    for i, edge in enumerate(ranked[:top_n], 1):
        name = id_to_canonical.get(edge.player_id, edge.player_id)
        mean_s = f"{edge.projected:.1f}"
        if edge.projected_stdev is not None:
            mean_s += f"±{edge.projected_stdev:.1f}"
        ev_s = f" EV={edge.best_ev:.3f}" if getattr(edge, "best_ev", None) is not None else ""
        odds_s = f" O{edge.over_odds}/U{edge.under_odds}" if (edge.over_odds is not None and edge.under_odds is not None) else ""
        book_s = f" ({edge.over_provider or '?'}/{edge.under_provider or '?'})" if (getattr(edge, "over_provider", None) or getattr(edge, "under_provider", None)) else ""
        res = projected.get((edge.player_id, edge.stat_type), (None, None, None))[1]
        confidence_s = res.confidence if (res and getattr(res, "confidence", None)) else "—"
        print(f"  {i}. {name} {edge.stat_type} {edge.market_line} | {mean_s}{ev_s} -> {edge.recommended_side}{odds_s}{book_s} | {confidence_s}")

    alert_top_n = _env_int("ALERT_TOP_N", 25)
    top_alerts = ranked[:alert_top_n]
    high_value_alerts = alert_high_value_props(
        top_alerts, min_edge=0.05, min_ev=0.05, player_names=id_to_canonical
    )
    print("Alerts sent for {} picks (Top {})".format(len(high_value_alerts), alert_top_n))

    def _above_alert_threshold(edge: Any) -> bool:
        if getattr(edge, "best_ev", None) is not None:
            return edge.best_ev >= 0.05
        return abs(getattr(edge, "edge", 0) or 0) >= 0.05

    save_all = _env_bool("SAVE_ALL", False)
    to_save = [e for e in ranked if _above_alert_threshold(e)] if save_all else high_value_alerts
    if to_save:
        db = DatabaseManager()
        try:
            for edge in to_save:
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
            print("Saved {} picks to the database.".format(len(to_save)))
        finally:
            db.close()


if __name__ == "__main__":
    asyncio.run(main())