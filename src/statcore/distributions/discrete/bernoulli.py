import numpy as np

from statcore.distributions.base import DiscreteDistribution


class Bernoulli(DiscreteDistribution):
    def __init__(self, p: float) -> None:
        if not 0 < p < 1:
            raise ValueError("Parameter/Variable p muss 0 < p < 1 entsprechen.")

        self.p = p

    def pmf(self, k: float) -> float:
        return np.power(self.p, k) * np.power((1 - self.p), (1 - k))

    def log_likelihood(self, data: np.ndarray) -> float:
        self.validate_data(data)
        p = np.clip(self.p, 1e-15, 1 - 1e-15)  # Buffer weil log(0) ergibt -inf

        l_p = np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))

        return l_p

    def mle_fit(self, data: np.ndarray) -> float:
        self.validate_data(data)

        p_hat = np.mean(
            data
        )  # unser Schätzer für Bernoulli selbst hergeleitet in der Mitschrift

        self.p = p_hat  # jetzt lernt Instanz aus den Daten

        return p_hat

    def validate_data(self, data: np.ndarray) -> None:
        super().validate_data(data)

        if not np.isin(data, [0, 1]).all():
            raise ValueError("Daten für die MLE müssen 0 oder 1 entsprechen.")
