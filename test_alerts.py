"""
Test/demo the alerting pipeline: rank props by edge, then send Telegram/Discord
alerts for every prop with edge > 5%.

Set env vars to receive on your phone:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (from @BotFather and a chat with the bot)
  and/or DISCORD_WEBHOOK_URL             (channel webhook)

Optional: PLAYER_NAMES as JSON map player_id -> display name for prettier alerts.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import MarketSnapshot, PropLine
from src.optimizer import rank_props_by_edge
from src.projection import StatType
from src.alerting import alert_high_value_props


def mock_projection(player_id: str, stat_type: StatType):
    """Example: LeBron over on points, others under so we get one high-value alert."""
    if player_id == "lebron":
        return 28.0   # line 25 -> edge 12%
    if player_id == "curry":
        return 3.0
    if player_id == "jokic":
        return 12.2
    return None


def main() -> None:
    snapshot = MarketSnapshot(
        snapshot_id="alert-run",
        game_id="game-1",
        lines=[
            PropLine(player_id="lebron", provider="TestBook", stat_type="Points", threshold=25.0),
            PropLine(player_id="curry", provider="TestBook", stat_type="Threes", threshold=5.0),
            PropLine(player_id="jokic", provider="TestBook", stat_type="Rebounds", threshold=12.0),
        ],
    )
    ranked = rank_props_by_edge(snapshot, mock_projection)
    player_names = {"lebron": "LeBron James", "curry": "Stephen Curry", "jokic": "Nikola Jokic"}

    high_value = alert_high_value_props(
        ranked,
        min_edge=0.05,
        player_names=player_names,
    )
    print(f"Alerts sent for {len(high_value)} high-value prop(s). Check Telegram/Discord if configured.")
    print("test_alerts.py: Alert pipeline completed successfully.")


if __name__ == "__main__":
    main()
