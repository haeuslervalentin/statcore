import numpy as np
import scipy

from statcore.distributions.base import DiscreteDistribution


class Poisson(DiscreteDistribution):
    def __init__(
        self,
        rate: float,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be bigger than zero.")

        self.rate = rate

    def validate_data(self, data: np.ndarray) -> None:
        """Validates the data np-array if it is suitable for the distribution."""
        super().validate_data(data)

        if np.any(data < 0):
            raise ValueError("data-values must be bigger than zero or zero.")

        if np.any(data % 1 != 0):
            raise ValueError("data-values must be integers.")

    def mle_fit(self, data: np.ndarray) -> float:
        self.validate_data(data)

        self.rate = np.mean(data)

        return float(self.rate)

    def pmf(self, k: np.ndarray) -> np.ndarray:
        """Calculates the Probability Mass Function (PDF) of the poisson-distribution."""
        self.validate_data(k)

        return np.exp(k * np.log(self.rate) - self.rate - scipy.special.gammaln(k + 1))
