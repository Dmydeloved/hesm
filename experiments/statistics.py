from __future__ import annotations

import math
import random
import statistics as py_statistics


def mean(values: list[float]) -> float:
    return py_statistics.fmean(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return float(ordered[index])


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    samples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(mean(sample))
    alpha = (1.0 - confidence) / 2.0
    return (percentile(draws, alpha), percentile(draws, 1.0 - alpha))


def paired_sign_test(left: list[float], right: list[float]) -> dict[str, float]:
    wins = sum(1 for l_value, r_value in zip(left, right) if l_value > r_value)
    losses = sum(1 for l_value, r_value in zip(left, right) if l_value < r_value)
    trials = wins + losses
    if trials == 0:
        return {"wins": 0.0, "losses": 0.0, "p_value": 1.0}
    tail = min(wins, losses)
    probability = sum(math.comb(trials, i) for i in range(tail + 1)) / (2**trials)
    return {"wins": float(wins), "losses": float(losses), "p_value": min(1.0, 2 * probability)}

