"""
PrimeProp main entrypoint: fetch PRE-GAME NBA odds from The Odds API,
run the optimizer, and print/alert the top +EV props.
"""
import os
import asyncio
import aiohttp
from datetime import datetime, timezone
from dotenv import load_dotenv

from src.models import Player
from src.database import DatabaseManager
from src.ingest import OddsApiIngestor, fetch_multi_source_snapshot
from src.optimizer import rank_props_by_edge
from src.projection import StatType
from src.alerting import alert_high_value_props

load_dotenv()

# Add a few players you know are playing in the UPCOMING games tonight/tomorrow!
PLAYERS = [
    Player(id="lebron", standardized_name="LeBron James", team="LAL", aliases=["LeBron", "LBJ"]),
    Player(id="wembanyama", standardized_name="Victor Wembanyama", team="SAS", aliases=["Wemby"]),
    Player(id="edwards", standardized_name="Anthony Edwards", team="MIN", aliases=["Ant"]),
    Player(id="durant", standardized_name="Kevin Durant", team="PHX", aliases=["KD"]),
    Player(id="luka", standardized_name="Luka Doncic", team="DAL", aliases=["Luka"]),
    Player(id="tatum", standardized_name="Jayson Tatum", team="BOS", aliases=["JT", "Tatum"]),
    Player(id="jokic", standardized_name="Nikola Jokic", team="DEN", aliases=["Joker"]),
]


def dummy_projection(player_id: str, stat_type: StatType) -> float:
    """Dummy projection: fixed values per stat type."""
    defaults: dict[str, float] = {
        "Points": 26.0,
        "Rebounds": 10.0,
        "Assists": 7.0,
        "PRA": 40.0,
        "Threes": 3.0,
    }
    return defaults.get(stat_type, 20.0)


async def get_upcoming_event_ids(api_key: str) -> list[str]:
    """Fetches games and returns IDs only for games that have NOT started yet."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    upcoming_ids = []
    now = datetime.now(timezone.utc)

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params={"apiKey": api_key}) as resp:
            games = await resp.json()

            for game in games:
                # API returns time like "2024-10-22T23:30:00Z", we make it Python-friendly
                commence_str = game["commence_time"].replace("Z", "+00:00")
                commence_time = datetime.fromisoformat(commence_str)

                # ONLY grab games where the tip-off time is in the future
                if commence_time > now:
                    upcoming_ids.append(game["id"])

    return upcoming_ids


async def main() -> None:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not set in environment. Add it to .env")

    print("Checking for upcoming pre-game matchups...")
    upcoming_event_ids = await get_upcoming_event_ids(api_key)
    if not upcoming_event_ids:
        print("No upcoming pre-game matchups found. All games for the day have already started!")
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
                },
            )
        )

    # Pass ALL the ingestors into your beautifully built concurrent snapshot fetcher
    snapshot = await fetch_multi_source_snapshot(
        snapshot_id="pre-game-1",
        game_id="upcoming_nba_slate",
        players=PLAYERS,
        providers=ingestors,
    )
    ranked = rank_props_by_edge(snapshot, dummy_projection)

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

    # Persist high-value picks to database
    db = DatabaseManager()
    player_names = {p.id: p.standardized_name for p in PLAYERS}
    for edge in high_value_alerts:
        player_name = player_names.get(edge.player_id, edge.player_id)
        db.log_pick(
            player_name=player_name,
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
