import numpy as np
import pytest
from scipy import stats

from statcore.distributions.discrete.binomial import Binomial


def test_binomial_pmf_correctness() -> None:
    """Test if the Binomial Probability Mass Function (PMF) comes near the scipy.stats"""
    p = 0.3
    n = 10
    dist = Binomial(p=p, n=n)
    k = np.array([7, 9, 4, 1, 1, 5, 5])

    np.testing.assert_allclose(
        actual=dist.pmf(k=k), desired=stats.binom.pmf(k, n, p), rtol=1e-6
    )


def test_binomial_mle_fit_converges_stably() -> None:
    n = 10
    p = 0.4
    dist = Binomial(p=p, n=n)

    random_gen = np.random.default_rng(seed=42)
    data = random_gen.binomial(n=n, p=p, size=100000)

    actual_p = dist.mle_fit(data=data)
    desired_p = np.mean(data) / n

    np.testing.assert_allclose(actual=actual_p, desired=desired_p, rtol=1e-2)


def test_binomial_validation_raises_error() -> None:
    with pytest.raises(ValueError):
        Binomial(p=-2, n=10)

    with pytest.raises(ValueError):
        Binomial(p=0.5, n=0)

    dist = Binomial(p=0.4, n=10)
    with pytest.raises(ValueError):
        dist.mle_fit(data=np.array([5, -1]))

    with pytest.raises(ValueError):
        dist.mle_fit(data=np.array([1, 1, 20]))

    with pytest.raises(ValueError):
        dist.mle_fit(data=np.array([0.58]))
