import numpy as np
import scipy

from statcore.distributions.base import DiscreteDistribution


class Binomial(DiscreteDistribution):
    def __init__(self, p: float, n: int) -> None:
        if p <= 0 or p >= 1:
            raise ValueError(
                "Parameter/Variable p must be in the open interval (0, 1)."
            )

        if n <= 0:
            raise ValueError("Parameter/Variable n must be bigger than zero.")

        self.p = p
        self.n = n

    def pmf(self, k: np.ndarray) -> np.ndarray:
        """Calculates the Probability Mass Function of the Binomial-Distribution"""
        self.validate_data(data=k)

        return np.exp(
            scipy.special.gammaln(self.n + 1)
            - scipy.special.gammaln(k + 1)
            - scipy.special.gammaln(self.n - k + 1)
            + k * np.log(self.p)
            + (self.n - k) * np.log(1 - self.p)
        )

    def validate_data(self, data: np.ndarray) -> None:
        """Validates the data given to check if it is suited for binomial."""
        super().validate_data(data=data)

        if not (np.all(0 <= data) and np.all(data <= self.n)):
            raise ValueError("data must be [0, n].")

        if np.any(data % 1 != 0):
            raise ValueError("data must not be a floating number.")

    def mle_fit(self, data: np.ndarray) -> float:
        """Calculates the parameter p for a given array of independent experiments."""

        self.validate_data(data=data)

        p: float = np.mean(data) / self.n
        self.p = p

        return self.p
