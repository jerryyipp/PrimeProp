from src.projection import get_projection

import os
import sys

# Ensure the src package is importable when running this script directly.
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from projection import StatType, get_projection


def main() -> None:
    historical_values = [10.0, 10.0, 10.0, 30.0]
    player_id = "mock_player"
    stat_type: StatType = "Points"

    simple_average = get_projection(
        player_id,
        stat_type,
        historical_values,
        n_games=4,
        method="simple_average",
    )
    weighted_average = get_projection(
        player_id,
        stat_type,
        historical_values,
        n_games=4,
        method="weighted_average",
    )

    print(f"Simple average: {simple_average}")
    print(f"Weighted average: {weighted_average}")

    assert simple_average == 15.0, f"Expected simple_average 15.0, got {simple_average}"
    assert weighted_average == 18.0, f"Expected weighted_average 18.0, got {weighted_average}"

    print("test_projection.py: All projection assertions passed successfully.")


if __name__ == "__main__":
    main()

