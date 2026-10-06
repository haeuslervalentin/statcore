import numpy as np

from statcore.distributions.base import ContinuousDistribution


class Normal(ContinuousDistribution):
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        if sigma <= 0:
            raise ValueError("sigma must be bigger than zero.")
        self.sigma = sigma

    def pdf(self, k: np.ndarray) -> np.ndarray:
        """Calculate the Probability density function (PDF) for an array of values"""
        self.validate_data(data=k)

        return (
            1
            / (np.sqrt(2 * np.pi * np.square(self.sigma)))
            * np.exp(-np.square(k - self.mu) / (2 * np.square(self.sigma)))
        )

    def log_likelihood(self, data: np.ndarray) -> float:
        """Calculates the Log-Likelihood of the given data under the current parameters."""
        self.validate_data(data=data)

        return len(data) * -0.5 * np.log(2 * np.pi * np.square(self.sigma)) + (
            -(np.sum(np.square(data - self.mu))) / (2 * np.square(self.sigma))
        )

    def mle_fit(self, data: np.ndarray) -> None:
        """Predicts the optimale parameters mu and sigma from the data (MLE) and updates the
        instance."""

        self.validate_data(data=data)

        mu = np.mean(data)
        self.mu = mu

        sigma = np.sqrt(np.sum(np.square(data - self.mu)) / len(data))
        self.sigma = sigma

    def validate_data(self, data: np.ndarray) -> None:
        """Validates, if the input data is suited for the normal-distribution"""
        super().validate_data(data=data)

        if np.any(np.isnan(data)):
            raise ValueError("data must not be nan.")

        if np.any(np.isinf(data)):
            raise ValueError("data values must not be inf or -inf.")
