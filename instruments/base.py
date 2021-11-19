"""
This module contains the fundamental building blocks of financial instruments.
"""

from enum import Enum
from abc import ABC, abstractmethod


class InstrumentLevel1(Enum):

    Cash = "Cash"
    Derivative = "Derivative"

    def __str__(self) -> str:
        return str(self.value)


class InstrumentLevel2(Enum):

    Equity = "Equity"
    Debt = "Debt"
    InterestRate = "InterestRate"
    Currency = "Currency"

    def __str__(self) -> str:
        return str(self.value)


class InstrumentLevel3(Enum):

    Share = "Share"
    Stock = "Stock"
    ZeroCouponBond = "ZeroCouponBond"
    FixedRateBond = "FixedRateBond"
    FloatingRateBond = "FloatingRateBond"
    EquityForward = "EquityForward"
    ForwardRateAgreement = "ForwardRateAgreement"
    CurrencyForward = "CurrencyForward"
    EuroDollarFuture = "EuroDollarFuture"
    EquityFuture = "EquityFuture"
    FixedLeg = "FixedLeg"
    FloatingLeg = "FloatingLeg"
    InterestRateSwap = "InterestRateSwap"

    def __str__(self) -> str:
        return str(self.value)


class Instrument(ABC):
    """
    A financial instrument is defined as any contract that gives rise
    to a financial asset of one entity and a financial liability or
    equity investment of another entity.
    """

    def __init__(
            self,
            quote_currency: str,
            instrument_level_1: InstrumentLevel1,
            instrument_level_2: InstrumentLevel2,
            instrument_level_3: InstrumentLevel3,
            tradeable: bool
    ) -> None:

        self.quote_currency = quote_currency
        self.instrument_level_1 = instrument_level_1
        self.instrument_level_2 = instrument_level_2
        self.instrument_level_3 = instrument_level_3
        self.tradeable = tradeable

    @abstractmethod
    def value(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def __repr__(self):
        pass
