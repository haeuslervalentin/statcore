import numpy as np

from statcore.distributions.base import DiscreteDistribution


class Geometric(DiscreteDistribution):
    def __init__(self, p: float) -> None:
        if not 0 < p < 1:
            raise ValueError("p must be in the open interval (0, 1).")

        self.p = p

    def validate_data(self, data: np.ndarray) -> None:
        super().validate_data(data)

        if np.any(data <= 0):
            raise ValueError("data values must be bigger than zero.")

        if np.any(data % 1 != 0):
            raise ValueError("data values must be values of integer.")

    def pmf(self, k: np.ndarray) -> np.ndarray:
        self.validate_data(data=k)

        log_pmf = (k - 1) * np.log(1 - self.p) + np.log(self.p)

        return np.exp(log_pmf)

    def log_likelihood(self, data: np.ndarray) -> float:
        self.validate_data(data=data)

        log_pmf = (data - 1) * np.log(1 - self.p) + np.log(self.p)

        return float(np.sum(log_pmf))

    def mle_fit(self, data: np.ndarray) -> float:
        self.validate_data(data=data)

        avg_of_data = np.average(data)

        p = 1.0 / avg_of_data

        self.p = p

        return float(self.p)
