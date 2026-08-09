import abc
from signal import raise_signal
from typing import Any, Type

import numpy as np


class BaseDistribution:
    """
    Abstrakte Basisklasse für alle Wahrscheinlichkeitsverteilungen in StatCore.
    Definiert die Kern-Schnittstelle für Likelihood-Berechnungen und MLE-Fitting.
    """

    @abc.abstractmethod
    def log_likelihood(self, data: np.ndarray) -> float:
        """
        Berechnet die Log-Likelihood der gegebenen Daten unter den aktuellen Parametern.
        Motivation: Nummerische Stabilität gegenüber der normalen Likelihood.
        """
        pass

    @abc.abstractmethod
    def mle_fit(self, data: np.ndarray) -> None:
        """
        Schätzt die Parameter der Verteilung basierend auf den Daten via Maximum Likelihood Estimation
        (MLE)
        """
        pass

    def validate_data(self, data: np.ndarray) -> None:
        """
        Basis-Validierung für Eingabedaten.
        Spezifische Verteilung sollten diese Methode erweitern (z.B. für Binärwerte)
        """

        if data.size == 0:
            raise ValueError("Eingabedaten dürfen nicht leer sein.")

        if not isinstance(data, np.ndarray):
            raise TypeError("Daten müssen als numpy.ndarray übergeben werden.")


class DiscreteDistribution(BaseDistribution):
    """
    Basisklasse für diskrete Verteilungen (Bernoulli, Poisson, etc.).
    """

    @abc.abstractmethod
    def pmf(self, k: np.ndarray) -> np.ndarray:
        """Berechnet die Wahrscheinlichkeitsfunktion (Probability Mass Function)."""

        pass


class ContinuousDistribution(BaseDistribution):
    """
    Basisklasse für stetige Verteilung (Normal, Exponential, etc.).
    """

    @abc.abstractmethod
    def pdf(self, k: np.ndarray) -> np.ndarray:
        """Berechnet die Wahrscheinlichkeitsdichtefuntion (Probability Density Function)"""
        pass
