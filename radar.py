import os
import asyncio
import aiohttp
from dotenv import load_dotenv

async def ping_radar():
    load_dotenv()
    api_key = os.getenv("ODDS_API_KEY")
    
    # Step 1: Get today's games (event IDs)
    events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    
    print("Fetching today's NBA games...")
    async with aiohttp.ClientSession() as session:
        async with session.get(events_url, params={"apiKey": api_key}) as resp:
            games = await resp.json()
            
            if not games:
                print("No games found today.")
                return
                
            game = games[0] # Let's just grab the first game on the list
            print(f"Found {len(games)} games. Checking the first one: {game['away_team']} @ {game['home_team']}")
            event_id = game['id']
            
            # Step 2: Get player props for that specific game
            props_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": "player_points,player_rebounds"
            }
            
            print(f"Fetching player props for event ID: {event_id}...")
            async with session.get(props_url, params=params) as props_resp:
                data = await props_resp.json()
                
                players = set()
                for book in data.get("bookmakers", []):
                    for market in book.get("markets", []):
                        for outcome in market.get("outcomes", []):
                            if name := outcome.get("description"): # 'description' usually holds the player name
                                players.add(name)
                
                print(f"\nFound {len(players)} players with live props in this game!")
                print("Here are a few:")
                print(list(players)[:10])

if __name__ == "__main__":
    asyncio.run(ping_radar())