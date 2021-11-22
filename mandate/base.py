"""
A set of rules defining trade objectives.
"""
from typing import Dict, List, Tuple
from exposures.base import Exposure

from instruments.portfolio import Portfolio
from economy.base import Economy


class Mandate:

    def __init__(self, exposures_and_targets: List[Tuple[Exposure, float]]) -> None:
        self.exposures_and_targets = exposures_and_targets

    def exposure_deviation(self, portfolio: Portfolio, economy: Economy) -> dict:
        deviation = {}
        for x in self.exposures_and_targets:
            exposure, exposure_target = x[0], x[1]
            exposure_portfolio = exposure.portfolio_exposure(portfolio, economy)
            deviation[exposure.identifier] = (exposure_portfolio-exposure_target)
        return deviation
