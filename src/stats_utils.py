"""Statistical helpers for two-variant A/B test analysis.

Convention: A = control, B = treatment. All effects are reported as B - A.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def srm_test(n_a: int, n_b: int, expected_ratio: float = 0.5) -> dict:
    """Sample Ratio Mismatch check (chi-square goodness of fit).

    A very small p-value (industry convention: < 0.001) suggests that
    randomisation or logging is broken and results should not be trusted.
    """
    total = n_a + n_b
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    chi2, p = stats.chisquare([n_a, n_b], f_exp=expected)
    return {"n_a": n_a, "n_b": n_b, "share_a": n_a / total, "chi2": chi2, "p_value": p}


def two_proportion_ztest(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Two-sided z-test for a difference in proportions, with a Wald CI."""
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "p_a": p_a,
        "p_b": p_b,
        "abs_diff": diff,
        "rel_diff": diff / p_a,
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "z": z,
        "p_value": p_value,
    }


def bootstrap_mean_diff(a, b, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap for the difference in means (B - A).

    Suited to heavily skewed metrics where a t-test's normality
    assumption is questionable.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(b, b.size).mean() - rng.choice(a, a.size).mean()
    return {
        "mean_diff": b.mean() - a.mean(),
        "ci_low": np.quantile(diffs, alpha / 2),
        "ci_high": np.quantile(diffs, 1 - alpha / 2),
        "samples": diffs,
    }


def bayesian_ab(x_a: int, n_a: int, x_b: int, n_b: int, n_draws: int = 200_000, seed: int = 42) -> dict:
    """Beta-Binomial model with a uniform Beta(1, 1) prior on each rate."""
    rng = np.random.default_rng(seed)
    post_a = rng.beta(1 + x_a, 1 + n_a - x_a, n_draws)
    post_b = rng.beta(1 + x_b, 1 + n_b - x_b, n_draws)
    diff = post_b - post_a
    return {
        "prob_b_better": (diff > 0).mean(),
        "expected_diff": diff.mean(),
        "cri_low": np.quantile(diff, 0.025),
        "cri_high": np.quantile(diff, 0.975),
        # expected loss (in rate points) if we ship B but A is actually better
        "expected_loss_b": np.maximum(-diff, 0).mean(),
        "expected_loss_a": np.maximum(diff, 0).mean(),
        "samples": diff,
    }


def mde_two_proportions(p_base: float, n_a: int, n_b: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute difference for a two-sided z-test."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    se = np.sqrt(p_base * (1 - p_base) * (1 / n_a + 1 / n_b))
    return (z_a + z_b) * se


def required_n_per_group(p_base: float, abs_diff: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Sample size per group needed to detect `abs_diff` from `p_base`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p2 = p_base + abs_diff
    var = p_base * (1 - p_base) + p2 * (1 - p2)
    return int(np.ceil((z_a + z_b) ** 2 * var / abs_diff**2))
