"""
A set of rules defining trade objectives.
"""
import numpy as np
from typing import Dict, List, Tuple
from exposures.base import Exposure

from instruments.portfolio import Portfolio
from economy.base import Economy


class Mandate:

    def __init__(self, exposures_and_targets: List[Tuple[Exposure, float]], instrument_generators) -> None:
        self.exposures_and_targets = exposures_and_targets
        self.instrument_generators = instrument_generators
        self.n_instruments = len(self.instrument_generators)
        # Todo: Incorporate in mandate reader.
        self.min_notional = -10000
        self.max_notional = +10000

    def exposure_deviations(self, portfolio: Portfolio, economy: Economy, as_array: bool = False) -> dict:
        deviation = {}
        for x in self.exposures_and_targets:
            exposure, exposure_target = x[0], x[1]
            exposure_portfolio = exposure.portfolio_exposure(portfolio, economy)
            deviation[exposure.identifier] = (exposure_portfolio-exposure_target)
        if as_array is True:
            deviation = np.array(list(deviation.values()))
        return deviation

    def abs_exposure_deviation(self, portfolio: Portfolio, economy: Economy) -> float:
        exposure_deviations = self.exposure_deviations(portfolio, economy, as_array=True)
        return np.sum(np.abs(exposure_deviations))
