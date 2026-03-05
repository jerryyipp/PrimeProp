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
from .projection import projection_interval


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

    Display: line, mean±stdev, p_model, best_ev, side, odds, book.
    """
    name = player_name if player_name is not None else prop_edge.player_id
    line_val = prop_edge.market_line
    mean_s = f"{prop_edge.projected:.1f}"
    if prop_edge.projected_stdev is not None:
        mean_s += f"±{prop_edge.projected_stdev:.1f}"
        try:
            level = float(os.getenv("PROJECTION_INTERVAL_LEVEL", "0.80").strip())
        except (ValueError, TypeError):
            level = 0.80
        level = max(0.5, min(0.999, level))
        low, high = projection_interval(prop_edge.projected, prop_edge.projected_stdev, level)
        mean_s += f" ({int(level * 100)}%: {low:.1f}–{high:.1f})"
    p_model = None
    if prop_edge.recommended_side == "Over" and prop_edge.p_over_model is not None:
        p_model = prop_edge.p_over_model
    elif prop_edge.recommended_side == "Under" and prop_edge.p_under_model is not None:
        p_model = prop_edge.p_under_model
    p_str = f"P(model)={p_model:.2f}" if p_model is not None else "P(model)=—"
    ev_str = f"best_ev={prop_edge.best_ev * 100:.2f}%" if prop_edge.best_ev is not None else "best_ev=—"
    side_str = prop_edge.recommended_side
    odds_str = f"{prop_edge.recommended_odds:+.0f}" if prop_edge.recommended_odds is not None else "—"
    book_str = prop_edge.recommended_provider or prop_edge.provider or "—"

    line_move_str = ""
    if (
        getattr(prop_edge, "open_line", None) is not None
        and getattr(prop_edge, "current_line", None) is not None
        and getattr(prop_edge, "delta_line", None) is not None
    ):
        d = prop_edge.delta_line
        line_move_str = f"\nLine moved: {prop_edge.open_line} -> {prop_edge.current_line} ({'+' if d >= 0 else ''}{d})"

    return (
        f"**High-value prop**\n"
        f"Player: {name}\n"
        f"Line: {prop_edge.stat_type} {line_val}\n"
        f"Proj: {mean_s}\n"
        f"{p_str} | {ev_str}\n"
        f"Side: {side_str} | Odds: {odds_str} | Book: {book_str}"
        f"{line_move_str}"
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
    min_ev: float = 0.05,
    player_names: Optional[Dict[str, str]] = None,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    discord_webhook_url: Optional[str] = None,
) -> List[PropEdge]:
    """
    Send notifications for props with positive EV or edge above threshold.

    When best_ev is available, filter by best_ev >= min_ev (default 5%).
    Otherwise fall back to |edge| > min_edge.

    Uses TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID and/or DISCORD_WEBHOOK_URL
    from the environment if not passed. Returns the list of props that
    were above threshold (for logging/callers).
    """
    token = telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    webhook = discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")

    def above_threshold(e: PropEdge) -> bool:
        if e.best_ev is not None:
            return e.best_ev >= min_ev
        return abs(e.edge) >= min_edge

    high_value = [e for e in ranked_edges if above_threshold(e)]
    names = player_names or {}

    for prop_edge in high_value:
        name = names.get(prop_edge.player_id)
        text = format_alert(prop_edge, name)
        if token and chat_id:
            send_telegram(text, token, chat_id)
        if webhook:
            send_discord(text, webhook)

    return high_value
