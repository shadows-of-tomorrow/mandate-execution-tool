import numpy as np
from typing import Dict, List, Tuple
from exposures.base import Exposure

from instruments.portfolio import Portfolio
from economy.base import Economy


class Mandate:

    def __init__(self, exposures_and_targets: List[Tuple[Exposure, float]], instrument_generators) -> None:
        self.exposures = [x[0] for x in exposures_and_targets]
        self.targets = [x[1] for x in exposures_and_targets]
        self.instrument_generators = instrument_generators
        self.n_instruments = len(self.instrument_generators)
        self.n_exposures = len(exposures_and_targets)

    def exposure_deviations(self, portfolio: Portfolio, economy: Economy, as_array: bool = False) -> dict:
        deviation = {}
        for k in range(len(self.exposures)):
            exposure, exposure_target = self.exposures[k], self.targets[k]
            exposure_portfolio = exposure.portfolio_exposure(portfolio, economy)
            deviation[exposure.identifier] = self._get_deviation(exposure_portfolio, exposure_target)
        if as_array is True:
            deviation = np.array(list(deviation.values())).reshape(-1, 1)
        return deviation

    def _get_deviation(self, exposure_portfolio, exposure_target):
        return np.clip((exposure_portfolio+1e-6) / (exposure_target+1e-6) - 1.0, -1.0, 1.0)