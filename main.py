"""
PrimeProp main entrypoint: fetch live NBA odds from The Odds API,
run the optimizer, and print the top +EV props.
"""

import os
import asyncio
import aiohttp  # noqa: F401 - used by ingest

from dotenv import load_dotenv

from src.models import Player
from src.ingest import OddsApiIngestor, fetch_multi_source_snapshot
from src.optimizer import rank_props_by_edge
from src.projection import StatType

load_dotenv()

PLAYERS = [
    Player(
        id="lebron",
        standardized_name="LeBron James",
        team="LAL",
        aliases=["LeBron", "LBJ", "King James"],
    ),
    Player(
        id="wembanyama",
        standardized_name="Victor Wembanyama",
        team="SAS",
        aliases=["Wemby", "Victor W", "Wembanyama"],
    ),
    Player(
        id="edwards",
        standardized_name="Anthony Edwards",
        team="MIN",
        aliases=["Ant", "Ant Man", "Anthony E", "AE"],
    ),
    Player(
        id="durant",
        standardized_name="Kevin Durant",
        team="PHX",
        aliases=["KD", "Kevin D", "Slim Reaper", "Durantula"],
    ),
    Player(
        id="luka",
        standardized_name="Luka Doncic",
        team="DAL",
        aliases=["Luka", "Luka Doncic", "The Don"],
    ),
]


def dummy_projection(player_id: str, stat_type: StatType) -> float:
    """Dummy projection: fixed values per stat type (projection provider doesn't receive threshold)."""
    defaults: dict[str, float] = {
        "Points": 26.0,
        "Rebounds": 10.0,
        "Assists": 7.0,
        "PRA": 40.0,
        "Threes": 3.0,
    }
    return defaults.get(stat_type, 20.0)


async def main() -> None:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not set in environment. Add it to .env")

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    ingestor = OddsApiIngestor(
        url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points,player_rebounds,player_assists",
        },
    )

    snapshot = await fetch_multi_source_snapshot(
        snapshot_id="live-1",
        game_id="basketball_nba",
        players=PLAYERS,
        providers=[ingestor],
    )

    ranked = rank_props_by_edge(snapshot, dummy_projection)

    print("Top 5 +EV bets from live data:")
    for i, edge in enumerate(ranked[:5], 1):
        pct = edge.edge * 100
        print(
            f"  {i}. {edge.player_id} | {edge.stat_type} {edge.recommended_side} {edge.market_line} | "
            f"Projected: {edge.projected} | Edge: {pct:.2f}% | {edge.provider}"
        )


if __name__ == "__main__":
    asyncio.run(main())
