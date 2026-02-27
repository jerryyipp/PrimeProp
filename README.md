# PrimeProp

Sports betting tool that helps optimize NBA player props by comparing your projections to live market lines, identifying +EV bets, and tracking performance over time.

## Features

- **Live odds ingestion** — Pulls pre-game player props from The Odds API (per-event endpoint) for upcoming NBA games
- **Fuzzy name matching** — Links player names across providers (e.g. "Steph Curry" ↔ "Stephen Curry")
- **Projection engine** — Baseline projections from last N games (weighted or simple average)
- **+EV optimizer** — Computes Edge = (Projected - Line) / Line, ranks props, recommends Over/Under/Pass
- **Real-time alerting** — Telegram and/or Discord alerts when |edge| > 5%
- **Persistence** — SQLite database logs every recommended pick for win-rate tracking (manual grading)

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```
   ODDS_API_KEY=your_api_key_here
   TELEGRAM_BOT_TOKEN=optional
   TELEGRAM_CHAT_ID=optional
   DISCORD_WEBHOOK_URL=optional
   ```

3. Get an API key from [The Odds API](https://the-odds-api.com/).

## Usage

- **Run the main pipeline** (fetch upcoming games, ingest props, rank, alert, persist):
  ```bash
  python main.py
  ```

- **Run tests** (no API calls, uses mock data):
  ```bash
  python test_projection.py
  python test_optimizer.py
  python test_database.py
  python test_alerts.py   # requires Telegram/Discord env vars for real alerts
  ```

- **Ping live players** (verify API and see who has odds posted):
  ```bash
  python radar.py
  ```

## Architecture

- `src/models.py` — Pydantic models (Player, PropLine, MarketSnapshot, Game)
- `src/ingest.py` — Async multi-source ingestor, FuzzyNameMatcher, OddsApiIngestor, PrizePicksIngestor
- `src/projection.py` — Historical stat projection (weighted/simple average)
- `src/optimizer.py` — Edge calculation, ranked PropEdge list, recommended_side
- `src/alerting.py` — Telegram/Discord notifications for high-value props
- `src/database.py` — SQLite persistence (log_pick, get_win_rate)
- `main.py` — Pre-game CLV pipeline: upcoming events → per-event ingestors → optimizer → alerts → database

## License

MIT
