import numpy as np
import pytest
from scipy import stats

from statcore.distributions.continuous.exponentail import Exponential


def test_exponential_pdf_correctness() -> None:
    """Compares the self implemented Normal Distributions Probability Density Function (PDF)
    with on from scipy.stats"""

    dist = Exponential(rate=0.5)
    k = np.array([4.0, 0.0, 1.0])

    actual = dist.pdf(k=k)
    desired = stats.expon.pdf(k, loc=0, scale=1.0 / dist.rate)

    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-6)


def test_exponential_mle_fit_converges_stably() -> None:
    """Tests if the mle_fit function konverts even noisy data stabel"""

    data_generator = np.random.default_rng(seed=42)
    data = np.array(data_generator.exponential(scale=2.0, size=10000))

    dist = Exponential(rate=1.0)

    _, desired_scale = stats.expon.fit(data, floc=0)
    desired_rate = 1.0 / desired_scale
    dist.mle_fit(data=data)

    actual_rate = dist.rate

    np.testing.assert_allclose(actual=actual_rate, desired=desired_rate, rtol=1e-6)


def test_exponential_validate_data_error_handling_correctness() -> None:
    with pytest.raises(ValueError):
        Exponential(rate=-4.5)

    with pytest.raises(ValueError):
        Exponential(rate=0.0)

    dist = Exponential(rate=0.5)

    with pytest.raises(ValueError):
        dist.validate_data(data=np.array([1.0, -0.5]))
