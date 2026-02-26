import os
import sys

# Ensure the src package is importable when running this script directly.
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from src.models import MarketSnapshot, PropLine
from src.optimizer import calculate_implied_probability, rank_props_by_edge
from src.projection import StatType


def mock_projection(player_id: str, stat_type: StatType):
    if player_id == "lebron":
        return 28.0
    if player_id == "curry":
        return 3.0
    if player_id == "jokic":
        return 12.2
    return None


def main() -> None:
    # Implied probabilities for sample American odds
    minus_110_prob = calculate_implied_probability(-110)
    plus_150_prob = calculate_implied_probability(150)
    print(f"Implied probability for -110: {minus_110_prob:.4f}")
    print(f"Implied probability for +150: {plus_150_prob:.4f}")

    snapshot = MarketSnapshot(
        snapshot_id="test-snapshot",
        game_id="game-1",
        lines=[
            PropLine(
                player_id="lebron",
                provider="TestBook",
                stat_type="Points",
                threshold=25.0,
            ),
            PropLine(
                player_id="curry",
                provider="TestBook",
                stat_type="Threes",
                threshold=5.0,
            ),
            PropLine(
                player_id="jokic",
                provider="TestBook",
                stat_type="Rebounds",
                threshold=12.0,
            ),
        ],
    )

    ranked = rank_props_by_edge(snapshot, mock_projection)

    print("\nRanked props by edge:")
    for edge in ranked:
        edge_pct = edge.edge * 100.0
        print(
            f"player_id={edge.player_id}, "
            f"stat_type={edge.stat_type}, "
            f"market_line={edge.market_line}, "
            f"projected={edge.projected}, "
            f"edge={edge_pct:.2f}%, "
            f"recommended_side={getattr(edge, 'recommended_side', 'N/A')}"
        )

    print("\ntest_optimizer.py: Completed ranking and printing edges.")


if __name__ == "__main__":
    main()

