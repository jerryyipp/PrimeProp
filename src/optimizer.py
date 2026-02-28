"""Expected Value (+EV) optimizer.

Builds comparison logic between model projections and market lines.

Edge formula:
    Edge = (Projected Stat - Market Line) / Market Line

The main entrypoint is `rank_props_by_edge`, which returns a ranked list
of the best-value props (highest positive edge first).
"""

from typing import Callable, List, Optional, Literal

from pydantic import BaseModel, Field

from .models import MarketSnapshot
from .projection import StatType


# Projections are supplied by the caller, typically backed by historical stats.
ProjectionProvider = Callable[[str, StatType], Optional[float]]


def calculate_implied_probability(odds: Optional[float]) -> Optional[float]:
    """
    Convert American odds (e.g. -110 or +125) to implied probability (0 to 1).

    Returns None if odds is None.
    """
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


class PropEdge(BaseModel):
    """Represents the value of a single prop relative to the model projection."""

    player_id: str = Field(..., description="Canonical Player.id for this prop")
    stat_type: StatType = Field(..., description="Stat category (points, rebounds, etc.)")
    provider: str = Field(..., description="Book/provider offering this prop")
    market_line: float = Field(..., gt=0, description="Posted line (threshold) from the book")
    projected: float = Field(..., description="Model's projected stat value")
    edge: float = Field(..., description="(projected - market_line) / market_line")
    recommended_side: Literal['Over', 'Under', 'Pass'] = Field(..., description="Recommended betting side: Over, Under, or Pass")


def compute_edge(projected: float, market_line: float) -> float:
    """
    Computes Edge = (projected - market_line) / market_line.

    Raises if market_line is not strictly positive.
    """
    if market_line <= 0.0:
        raise ValueError("market_line must be > 0 to compute edge.")
    return (projected - market_line) / market_line


def rank_props_by_edge(
    snapshot: MarketSnapshot,
    get_projection: ProjectionProvider,
) -> List[PropEdge]:
    """
    Returns a ranked list of props with their Edge, best value first.

    Args:
        snapshot: MarketSnapshot containing the available prop lines.
        get_projection: Callable that returns a projected stat value for a
            given (player_id, stat_type). If it returns None, that prop is
            skipped because there is no projection.
    """
    ranked: List[PropEdge] = []

    for line in snapshot.lines:
        projected = get_projection(line.player_id, line.stat_type)
        if projected is None:
            continue

        try:
            edge_value = compute_edge(projected, line.threshold)
        except ValueError:
            # Skip malformed or invalid lines (e.g. non-positive thresholds).
            continue

        if edge_value > 0.05:
            recommended_side = 'Over'
        elif edge_value < -0.05:
            recommended_side = 'Under'
        else:
            recommended_side = 'Pass'

        ranked.append(
            PropEdge(
                player_id=line.player_id,
                stat_type=line.stat_type,
                provider=line.provider,
                market_line=line.threshold,
                projected=projected,
                edge=edge_value,
                recommended_side=recommended_side,
            )
        )

    ranked.sort(key=lambda item: item.edge, reverse=True)
    return ranked

