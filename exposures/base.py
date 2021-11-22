from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy

from instruments.base import InstrumentLevel1, InstrumentLevel2
from instruments.portfolio import Portfolio
from economy.base import Economy


class ExposureType(Enum):
    AssetAllocationEquity = "AssetAllocationEquity"
    AssetAllocationDebt = "AssetAllocationDebt"
    ZeroDelta = "ZeroDelta"


class Exposure(ABC):

    def __init__(self, identifier: str, exposure_type: ExposureType) -> None:
        self.identifier = identifier
        self.exposure_type = exposure_type

    @abstractmethod
    def portfolio_exposure(self, portfolio: Portfolio, economy: Economy) -> float:
        # Todo: Re-use portfolio calculations.
        pass


class AssetAllocationEquity(Exposure):

    def __init__(self, identifier: str) -> None:
        super().__init__(identifier=identifier, exposure_type=ExposureType.AssetAllocationEquity)

    def portfolio_exposure(self, portfolio: Portfolio, economy: Economy) -> float:
        portfolio_equity = portfolio.filter_on_level_2([InstrumentLevel2.Equity])
        return portfolio_equity.value(economy)


class AssetAllocationDebt(Exposure):

    def __init__(self, identifier: str) -> None:
        super().__init__(identifier=identifier, exposure_type=ExposureType.AssetAllocationDebt)

    def portfolio_exposure(self, portfolio: Portfolio, economy: Economy) -> float:
        portfolio_debt = portfolio.filter_on_level_2([InstrumentLevel2.Debt])
        return portfolio_debt.value(economy)


class ZeroDelta(Exposure):

    def __init__(self, identifier: str, curve_identifier: str, tenor: float, bump_size: float = 0.0001) -> None:
        super().__init__(identifier=identifier, exposure_type=ExposureType.ZeroDelta)
        self.curve_identifier = curve_identifier
        self.tenor = tenor
        self.bump_size = bump_size

    def portfolio_exposure(self, portfolio: Portfolio, economy: Economy) -> float:
        return self._compute_delta(portfolio, economy, self.tenor)

    def _compute_delta(self, portfolio: Portfolio, economy: Economy, tenor: float) -> float:
        economy_up = deepcopy(economy)
        economy_down = deepcopy(economy)
        economy_up.yield_curves[self.curve_identifier].bump_tenor(tenor, self.bump_size)
        economy_down.yield_curves[self.curve_identifier].bump_tenor(tenor, bump_size=-self.bump_size)
        return (portfolio.value(economy_up)-portfolio.value(economy_down))/(2.0*self.bump_size)