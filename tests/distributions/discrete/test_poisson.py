import numpy as np
import pytest
from scipy import stats

from statcore.distributions.discrete.poisson import Poisson


def test_poisson_pmf_correctness() -> None:
    """Test if the Poisson Probability Mass Function (PMF) comes near the scipy.stats"""

    dist = Poisson(rate=1.5)
    k = np.array([7, 9, 4, 1, 1, 5, 5])

    np.testing.assert_allclose(
        actual=dist.pmf(k), desired=stats.poisson.pmf(k=k, mu=1.5), rtol=1e-6
    )


def test_poisson_mle_fit_converges_stably() -> None:
    dist = Poisson(rate=1.5)

    random_gen = np.random.default_rng(seed=42)
    data = random_gen.poisson(lam=1.5, size=100000)

    actual_rate = dist.mle_fit(data)
    desired_rate = np.mean(data)

    np.testing.assert_allclose(actual=actual_rate, desired=desired_rate, rtol=1e-6)


def test_poisson_validation_raises_error() -> None:
    with pytest.raises(ValueError):
        Poisson(rate=0)

    with pytest.raises(ValueError):
        Poisson(rate=-1.5)

    dist = Poisson(rate=1.5)

    with pytest.raises(ValueError):
        dist.validate_data(data=np.array([5, -1]))

    with pytest.raises(ValueError):
        dist.validate_data(data=np.array([5, 2.75]))
