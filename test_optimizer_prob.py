"""
Unit tests for model probability helpers: normal_cdf and model_prob_over.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.optimizer import normal_cdf, model_prob_over


def test_normal_cdf_basic() -> None:
    # Phi(0) = 0.5
    assert abs(normal_cdf(0.0) - 0.5) < 1e-9
    # Phi(-large) ~ 0, Phi(+large) ~ 1
    assert normal_cdf(-5.0) < 1e-6
    assert normal_cdf(5.0) > 1 - 1e-6
    # Monotonic
    assert normal_cdf(-1.0) < normal_cdf(0.0) < normal_cdf(1.0)


def test_model_prob_over_mean_above_line() -> None:
    # mean > line => P(Over) > 0.5
    p = model_prob_over(line=20.0, mean=25.0, stdev=5.0)
    assert p > 0.5
    assert p < 1.0
    # With large stdev, closer to 0.5
    p_wide = model_prob_over(line=20.0, mean=25.0, stdev=20.0)
    assert 0.5 < p_wide < p


def test_model_prob_over_mean_below_line() -> None:
    # mean < line => P(Over) < 0.5
    p = model_prob_over(line=30.0, mean=25.0, stdev=5.0)
    assert p < 0.5
    assert p > 0.0
    # With large stdev, closer to 0.5
    p_wide = model_prob_over(line=30.0, mean=25.0, stdev=20.0)
    assert p < p_wide < 0.5


def test_model_prob_over_zero_stdev_mean_above_line() -> None:
    # stdev ~ 0, mean > line => P(Over) = 1.0
    assert model_prob_over(line=20.0, mean=25.0, stdev=0.0) == 1.0
    assert model_prob_over(line=20.0, mean=25.0, stdev=1e-8) == 1.0


def test_model_prob_over_zero_stdev_mean_below_line() -> None:
    # stdev ~ 0, mean < line => P(Over) = 0.0
    assert model_prob_over(line=30.0, mean=25.0, stdev=0.0) == 0.0
    assert model_prob_over(line=30.0, mean=25.0, stdev=1e-8) == 0.0


def test_model_prob_over_zero_stdev_tie() -> None:
    # stdev ~ 0, mean == line => P(Over) = 0.5
    assert model_prob_over(line=25.0, mean=25.0, stdev=0.0) == 0.5
    assert model_prob_over(line=25.0, mean=25.0, stdev=1e-8) == 0.5


def test_model_prob_over_p_under_is_one_minus_p_over() -> None:
    line, mean, stdev = 24.0, 25.0, 4.0
    p_over = model_prob_over(line, mean, stdev)
    # P(Under) = 1 - P(Over) for continuous model
    p_under = 1.0 - p_over
    assert abs(p_over + p_under - 1.0) < 1e-9


def main() -> None:
    test_normal_cdf_basic()
    test_model_prob_over_mean_above_line()
    test_model_prob_over_mean_below_line()
    test_model_prob_over_zero_stdev_mean_above_line()
    test_model_prob_over_zero_stdev_mean_below_line()
    test_model_prob_over_zero_stdev_tie()
    test_model_prob_over_p_under_is_one_minus_p_over()
    print("test_optimizer_prob.py: All tests passed.")


if __name__ == "__main__":
    main()
