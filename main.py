"""
PrimeProp main entrypoint: fetch PRE-GAME NBA odds from The Odds API,
run the optimizer, and print/alert the top +EV props.
"""
import os
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv()  # MUST BE LOADED BEFORE SRC MODULES

from src.database import DatabaseManager
from src.ingest import OddsApiIngestor, fetch_multi_source_snapshot
from src.optimizer import rank_props_by_edge
from src.projection import StatType, get_projection
from src.alerting import alert_high_value_props
from src.stats import fetch_last_n_game_values


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
                    "markets": "player_points,player_rebounds,player_assists",
                    "oddsFormat": "american",
                },
            )
        )

    # Pass ALL the ingestors into your beautifully built concurrent snapshot fetcher
    snapshot = await fetch_multi_source_snapshot(
        snapshot_id="pre-game-1",
        game_id="upcoming_nba_slate",
        players=[],
        providers=ingestors,
    )

    print(f"DEBUG: Total props ingested: {len(snapshot.lines)}")

    # Build projections from real historical stats (last 10 games, weighted average)
    print("Fetching historical stats and building projections...")
    projections: dict[tuple[str, StatType], float] = {}
    unique_keys = {(line.player_id, line.stat_type) for line in snapshot.lines}

    print(f"DEBUG: Unique (player, stat_type) pairs found: {len(unique_keys)}")

    total_pairs = len(unique_keys)
    async with aiohttp.ClientSession() as stats_session:
        for i, (player_id, stat_type) in enumerate(unique_keys, 1):
            print(f"[{i}/{total_pairs}] Fetching stats for {player_id} ({stat_type}).")
            values = await fetch_last_n_game_values(
                player_name=player_id,
                stat_type=stat_type,
                n_games=10,
                session=stats_session,
            )
            if not values:
                continue
            proj = get_projection(
                player_id=player_id,
                stat_type=stat_type,
                historical_values=values,
                n_games=10,
                method="weighted_average",
            )
            projections[(player_id, stat_type)] = proj

    print(f"DEBUG: Successfully computed projections for {len(projections)} pairs.")

    def projection_provider(player_id: str, stat_type: StatType) -> float | None:
        return projections.get((player_id, stat_type))

    ranked = rank_props_by_edge(snapshot, projection_provider)

    print("\nTop 5 PRE-GAME +EV bets:")
    for i, edge in enumerate(ranked[:5], 1):
        pct = edge.edge * 100
        print(
            f"  {i}. {edge.player_id} | {edge.stat_type} {edge.recommended_side} {edge.market_line} | "
            f"Projected: {edge.projected} | Edge: {pct:.2f}% | {edge.provider}"
        )

    # Fire off alerts
    high_value_alerts = alert_high_value_props(ranked, min_edge=0.05)
    print(f"\nSuccessfully fired alerts for {len(high_value_alerts)} high-value pre-game props!")

    # Persist high-value picks to database (only if we have any)
    if high_value_alerts:
        db = DatabaseManager()
        for edge in high_value_alerts:
            db.log_pick(
                player_name=edge.player_id,
                stat_type=edge.stat_type,
                market_line=edge.market_line,
                projected=edge.projected,
                edge=edge.edge,
                recommended_side=edge.recommended_side,
            )
        print(f"Saved {len(high_value_alerts)} picks to the database.")
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
