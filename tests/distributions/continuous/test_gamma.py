import numpy as np
import pytest
from scipy import stats

from statcore.distributions.continuous.gamma import Gamma


def test_gamma_pdf_correctness() -> None:
    """Compares the self implemented Normal Distributions Probability Density Function (PDF)
    with on from scipy.stats"""

    dist = Gamma(alpha=1.5, beta=2.45)
    k = np.array([4.0, 2.0, 1.0])

    actual = dist.pdf(k=k)
    desired = stats.gamma.pdf(k, a=dist.alpha, loc=0, scale=1.0 / dist.beta)

    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-6)


def test_gamma_mle_fit_converges_stably() -> None:
    """Tests if the mle_fit function konverts even noisy data stabel"""
    dist = Gamma(alpha=1.0, beta=1.0)  # init with different values to check correctness

    data_generator = np.random.default_rng(seed=42)
    data = np.array(data_generator.gamma(shape=1.5, scale=1.0 / 2.45, size=10000))

    desired_alpha, _, desired_scale = stats.gamma.fit(data, floc=0)
    desired_beta = 1.0 / desired_scale

    dist.mle_fit(data=data)

    actual_alpha = dist.alpha
    actual_beta = dist.beta

    np.testing.assert_allclose(actual=actual_alpha, desired=desired_alpha, rtol=1e-6)
    np.testing.assert_allclose(actual=actual_beta, desired=desired_beta, rtol=1e-6)


def test_gamma_validate_data_error_handling_correctness() -> None:
    with pytest.raises(ValueError):
        Gamma(alpha=-1.0, beta=2.45)

    with pytest.raises(ValueError):
        Gamma(alpha=0.0, beta=2.45)

    with pytest.raises(ValueError):
        Gamma(alpha=1.0, beta=-2.45)

    with pytest.raises(ValueError):
        Gamma(alpha=1.0, beta=0.0)

    with pytest.raises(ValueError):
        dist = Gamma(alpha=1.0, beta=0.0)

        dist.validate_data(np.array([1.0, 0.0]))

    with pytest.raises(ValueError):
        dist = Gamma(alpha=1.0, beta=0.0)

        dist.validate_data(np.array([1.0, -2.0]))
