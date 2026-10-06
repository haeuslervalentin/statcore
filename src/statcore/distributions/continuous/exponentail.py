import numpy as np

from statcore.distributions.base import ContinuousDistribution


class Exponential(ContinuousDistribution):
    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("sigma must be bigger than zero.")

        self.rate = rate

    def validate_data(self, data: np.ndarray) -> None:
        super().validate_data(data)

        if np.any(data < 0):
            raise ValueError("data must be zero or bigger.")

    def mle_fit(self, data: np.ndarray) -> float:
        self.validate_data(data)

        self.rate = 1.0 / np.mean(data)

        return float(self.rate)

    def log_likelihood(self, data: np.ndarray) -> float:
        self.validate_data(data)

        log_pdf = np.log(self.rate) - self.rate * data

        return float(np.sum(log_pdf))

    def pdf(self, k: np.ndarray) -> np.ndarray:
        """Probability Density Function"""

        self.validate_data(k)

        log_pdf = np.log(self.rate) - self.rate * k

        return np.exp(log_pdf)
