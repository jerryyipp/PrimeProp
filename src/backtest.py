"""
Backtest graded picks: hit rate, Brier score, log loss, calibration bin report (avg predicted vs actual hit rate),
EV vs realized profit. Use --fit to fit and save calibration (a, b).
"""

import math
from pathlib import Path
from typing import Optional

from .database import DatabaseManager
from .optimizer import profit_per_unit


# Bins for model probability (Over): (low, high) in [0, 1]
P_OVER_BINS = [
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 1.00),
]


def _p_win_for_pick(row) -> Optional[float]:
    """Model P(win) for this pick: p_over if Over, 1-p_over if Under."""
    try:
        p_over = row["p_over_model"] if "p_over_model" in row.keys() else None
    except (KeyError, TypeError):
        p_over = None
    if p_over is None:
        return None
    side = (row.get("recommended_side") or "").strip()
    if side == "Over":
        return float(p_over)
    if side == "Under":
        return 1.0 - float(p_over)
    return None


def run_backtest(db_path: Optional[Path] = None) -> None:
    """
    Load graded picks from SQLite and print:
    - overall hit rate
    - hit rate by p_over_model bins (0.50-0.55, ... 0.75-0.80)
    - EV vs realized profit (per unit)
    """
    db = DatabaseManager(db_path=db_path)
    rows = db.get_graded_picks()
    db.close()

    if not rows:
        print("No graded picks (won IS NOT NULL). Grade picks to run backtest.")
        return

    n = len(rows)
    wins = sum(1 for r in rows if r["won"] == 1)
    hit_rate_pct = round(wins / n * 100.0, 2)
    print("--- Backtest (graded picks only) ---")
    print(f"Overall: n={n}, wins={wins}, hit rate={hit_rate_pct}%")

    # Brier score and log loss (picks with p_win)
    p_win_outcomes: list[tuple[float, int]] = []
    for r in rows:
        p_win = _p_win_for_pick(r)
        if p_win is None:
            continue
        outcome = 1 if r["won"] == 1 else 0
        p_win_outcomes.append((p_win, outcome))
    if p_win_outcomes:
        n_prob = len(p_win_outcomes)
        brier = sum((p - y) ** 2 for p, y in p_win_outcomes) / n_prob
        eps = 1e-15
        log_loss = -sum(
            y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
            for p, y in p_win_outcomes
        ) / n_prob
        print("\nBrier score (lower better): {:.4f}".format(brier))
        print("Log loss (lower better): {:.4f}".format(log_loss))

    # Calibration bin report: avg predicted vs actual hit rate
    print("\nCalibration bins (avg predicted vs actual hit rate):")
    for low, high in P_OVER_BINS:
        in_bin = []
        for r in rows:
            p_win = _p_win_for_pick(r)
            if p_win is None:
                continue
            if low <= p_win < high:
                in_bin.append(r)
        if not in_bin:
            print(f"  {low:.2f}-{high:.2f}: (no picks)")
            continue
        avg_pred = sum(_p_win_for_pick(r) for r in in_bin) / len(in_bin)
        b_wins = sum(1 for r in in_bin if r["won"] == 1)
        actual_hr = b_wins / len(in_bin)
        print(f"  {low:.2f}-{high:.2f}: n={len(in_bin)}, avg_predicted={avg_pred:.3f}, actual_hit_rate={actual_hr:.3f}")

    # Expected EV (model) vs realized profit (actual), per $1 stake
    total_ev = 0.0
    total_realized = 0.0
    used = 0
    for r in rows:
        p_win = _p_win_for_pick(r)
        try:
            odds = r["odds"] if "odds" in r.keys() else None
        except (KeyError, TypeError):
            odds = None
        if p_win is None or odds is None:
            continue
        profit = profit_per_unit(odds)
        if profit is None:
            continue
        total_ev += p_win * profit - (1.0 - p_win) * 1.0
        won = 1 if r["won"] == 1 else 0
        total_realized += won * profit - (1 - won) * 1.0
        used += 1

    print("\nExpected EV (model) vs realized profit (actual), per $1 stake:")
    print(f"  Picks used: {used} / {n}")
    print(f"  Sum expected EV:   {total_ev:.4f}")
    print(f"  Sum realized:      {total_realized:.4f}")
    print(f"  Difference (EV − realized): {total_ev - total_realized:.4f}")


if __name__ == "__main__":
    import sys
    from .optimizer import fit_calibration_from_picks, save_calibration, DEFAULT_CALIBRATION_PATH

    if "--fit" in sys.argv:
        db = DatabaseManager()
        rows = db.get_graded_picks()
        db.close()
        if len(rows) < 10:
            print("Need at least 10 graded picks to fit calibration. Run backtest without --fit first.")
        else:
            a, b = fit_calibration_from_picks(rows)
            save_calibration(DEFAULT_CALIBRATION_PATH, a, b)
            print(f"Fitted calibration a={a:.4f}, b={b:.4f} -> saved to {DEFAULT_CALIBRATION_PATH}")
    run_backtest()
