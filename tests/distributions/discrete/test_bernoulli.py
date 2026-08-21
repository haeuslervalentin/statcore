import numpy as np
import pytest
from scipy import stats

from statcore.distributions.discrete.bernoulli import Bernoulli


def test_bernoulli_pmf_correctness() -> None:
    """Überprüft die Wahrscheinlichkeitsfunktion (PMF) gegen scipy.stats."""
    p = 0.3
    dist = Bernoulli(p)

    np.testing.assert_allclose(dist.pmf(np.array([1.0])), p)
    np.testing.assert_allclose(dist.pmf(np.array([0.0])), 1.0 - p)

    k = np.array([11])
    expected = stats.bernoulli.pmf(k, p)
    actual = dist.pmf(k)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_bernoulli_mle_fit_synthetic_data() -> None:
    """Überprüft, ob der MLE-Schätzer den wahren Parameter approximiert [10]."""
    true_p = 0.7
    rng = np.random.default_rng(42)
    synthetic_data = rng.binomial(1, true_p, size=10000)

    dist = Bernoulli(p=0.5)

    estimated_p = dist.mle_fit(synthetic_data)

    assert np.isclose(estimated_p, true_p, atol=1e-2)
    assert np.isclose(dist.p, estimated_p)


def test_bernoulli_validation_raise_error() -> None:
    """Überprüft, ob bei ungültigen Eingaben (Daten außerhalb von {0, 1})
    ein ValueError geworfen wird."""

    dist = Bernoulli(p=0.5)
    with pytest.raises(ValueError):
        dist.validate_data(np.array([5, 6]))
