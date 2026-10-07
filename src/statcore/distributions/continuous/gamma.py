import numpy as np
from scipy.special import digamma, gammaln, polygamma

from statcore.distributions.base import ContinuousDistribution


class Gamma(ContinuousDistribution):
    def __init__(self, alpha: float, beta: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be bigger than zero.")

        if beta <= 0:
            raise ValueError("beta must be bigger than zero.")

        self.alpha = alpha
        self.beta = beta

    def validate_data(self, data: np.ndarray) -> None:
        super().validate_data(data)

        if np.any(data <= 0):
            raise ValueError("data must be bigger than zero.")

    def mle_fit(self, data: np.ndarray) -> tuple[float, float]:
        self.validate_data(data=data)

        mean_value = np.mean(data)
        mean_log_data = np.mean(np.log(data))
        var_value = np.var(data)

        alpha = (mean_value**2) / (var_value)

        for _ in range(100):
            delta = (
                np.log(alpha) - np.log(mean_value) - digamma(alpha) + mean_log_data
            ) / (1.0 / alpha - polygamma(1, alpha))

            alpha = alpha - delta

            if abs(delta) < 1e-7:
                break

        beta = alpha / mean_value
        self.alpha = alpha
        self.beta = beta

        return (float(self.alpha), float(self.beta))

    def log_likelihood(self, data: np.ndarray) -> float:
        self.validate_data(data=data)

        log_pdf = (
            self.alpha * np.log(self.beta)
            - gammaln(self.alpha)
            + (self.alpha - 1.0) * np.log(data)
            - self.beta * data
        )

        return float(np.sum(log_pdf))

    def pdf(self, k: np.ndarray) -> np.ndarray:
        self.validate_data(data=k)
        log_pdf = (
            self.alpha * np.log(self.beta)
            - gammaln(self.alpha)
            + (self.alpha - 1.0) * np.log(k)
            - self.beta * k
        )

        return np.exp(log_pdf)
