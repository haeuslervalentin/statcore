import numpy as np
import pytest
from scipy import stats

from statcore.distributions.discrete.geometric import Geometric


def test_geometric_pmf_correctness() -> None:
    k = np.array([1, 2, 3, 5, 8])
    p = 0.3
    dist = Geometric(p=0.3)

    actual_pmf_value = dist.pmf(k=k)
    desired_pmf_value = stats.geom.pmf(k, p)

    np.testing.assert_allclose(
        actual=actual_pmf_value, desired=desired_pmf_value, rtol=1e-6
    )


def test_geometric_mle_fit_converges_stably() -> None:
    p = 0.35
    data = np.random.default_rng(42).geometric(p=p, size=100000)
    dist = Geometric(p=p)

    actual_mle_fit_value = dist.mle_fit(data=data)
    desired_mle_fit_value = p

    np.testing.assert_allclose(
        actual=actual_mle_fit_value, desired=desired_mle_fit_value, rtol=1e-2
    )


def test_geometric_validation_raises_error() -> None:
    with pytest.raises(ValueError):
        Geometric(p=0)

    with pytest.raises(ValueError):
        Geometric(p=-1.5)

    with pytest.raises(ValueError):
        Geometric(p=5.5)

    dist = Geometric(p=0.5)

    with pytest.raises(ValueError):
        dist.validate_data(data=np.array([5, -1]))

    with pytest.raises(ValueError):
        dist.validate_data(data=np.array([5, 2.75]))
