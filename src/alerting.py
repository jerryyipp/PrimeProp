"""
Real-time alerting for high-value props (Edge > 5%).

Sends formatted notifications to Telegram and/or Discord when the optimizer
identifies props with edge above the threshold. Alerts include Player Name,
Prop Line, and Confidence Score.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .optimizer import PropEdge


# Confidence Score = edge as a percentage (e.g. 7.5 for 7.5% edge).
def confidence_score(edge: float) -> float:
    """Convert edge (decimal) to a percentage confidence score."""
    return round(edge * 100.0, 2)


def format_alert(
    prop_edge: PropEdge,
    player_name: Optional[str] = None,
) -> str:
    """
    Format a single high-value prop as an alert message.

    Includes: Player Name (or player_id), Prop Line (stat type, line, side),
    and Confidence Score.
    """
    name = player_name if player_name is not None else prop_edge.player_id
    score = confidence_score(prop_edge.edge)
    line_desc = (
        f"{prop_edge.stat_type} {prop_edge.recommended_side} {prop_edge.market_line}"
    )
    return (
        f"**High-value prop**\n"
        f"Player: {name}\n"
        f"Prop: {line_desc}\n"
        f"Projected: {prop_edge.projected} | Line: {prop_edge.market_line}\n"
        f"Confidence Score: {score}%\n"
        f"Provider: {prop_edge.provider}"
    )


def send_telegram(
    message: str,
    bot_token: str,
    chat_id: str,
) -> bool:
    """
    Send a text message via the Telegram Bot API.

    Requires bot_token (from @BotFather) and chat_id (your chat with the bot).
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def send_discord(
    message: str,
    webhook_url: str,
) -> bool:
    """
    Send a message to a Discord channel via webhook.

    Create a webhook in Discord: Channel → Edit → Integrations → Webhooks.
    """
    data = json.dumps({"content": message}).encode()
    req = urllib.request.Request(webhook_url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except (urllib.error.URLError, OSError):
        return False


def alert_high_value_props(
    ranked_edges: List[PropEdge],
    *,
    min_edge: float = 0.05,
    player_names: Optional[Dict[str, str]] = None,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    discord_webhook_url: Optional[str] = None,
) -> List[PropEdge]:
    """
    Send notifications for every prop with |edge| > min_edge (default 5%).

    Alerts on both profitable Overs (edge > 0.05) and profitable Unders
    (edge < -0.05); negative edge means the model projects under the line,
    so the Under is the recommended side.

    Uses TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID and/or DISCORD_WEBHOOK_URL
    from the environment if not passed. Returns the list of props that
    were above threshold (for logging/callers).
    """
    token = telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    webhook = discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")

    high_value = [e for e in ranked_edges if abs(e.edge) > min_edge]
    names = player_names or {}

    for prop_edge in high_value:
        name = names.get(prop_edge.player_id)
        text = format_alert(prop_edge, name)
        if token and chat_id:
            send_telegram(text, token, chat_id)
        if webhook:
            send_discord(text, webhook)

    return high_value
