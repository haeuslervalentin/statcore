import numpy as np
import pytest
from scipy import stats

from statcore.distributions.continuous.normal import Normal


def test_normal_pdf_correctness() -> None:
    """Compares the self implemented Normal Distributions Probability Density Function (PDF)
    with on from scipy.stats"""

    dist = Normal(mu=0.75, sigma=1.5)
    k = np.array([-1.0, 0.0, 1.0])

    actual = dist.pdf(k=k)
    desired = stats.norm.pdf(k, loc=0.75, scale=1.5)

    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-6)


def test_normal_mle_fit_converges_stably() -> None:
    """Tests if the mle_fit function konverts even noisy data stabel"""

    data_generator = np.random.default_rng()
    data = np.array(data_generator.normal(loc=0.75, scale=1.5, size=10000))

    dist = Normal(mu=0.75, sigma=1.5)

    desired_mu, desired_sigma = stats.norm.fit(data)
    dist.mle_fit(data=data)

    actual_mu = dist.mu
    actual_sigma = dist.sigma

    np.testing.assert_allclose(actual=actual_mu, desired=desired_mu, rtol=1e-6)
    np.testing.assert_allclose(actual=actual_sigma, desired=desired_sigma, rtol=1e-6)


def test_normal_validate_data_error_handling_correctness() -> None:
    dist = Normal(mu=0.75, sigma=1.5)
    nan_data = np.array([0.5, 0.0, np.nan, 1.0])
    inf_data = np.array([0.5, 0.0, np.inf, 1.0])

    with pytest.raises(ValueError):
        to_small_sigma = Normal(mu=0.75, sigma=-0.75)

    with pytest.raises(ValueError):
        dist.validate_data(data=nan_data)

    with pytest.raises(ValueError):
        dist.validate_data(data=inf_data)
