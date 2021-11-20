from typing import Union
from enum import Enum
from datetime import datetime
from abc import ABCMeta, abstractmethod

from economy.observables.base import Observable
from economy.observables.interest_rate import InterestRate
from instruments.base import Instrument, InstrumentLevel2, InstrumentLevel3
from instruments.cash.debt import FixedLeg, FloatingLeg
from instruments.derivatives.base import DerivativeInstrument
from economy.base import Economy
from economy.term_structures.yield_curve import YieldCurve


class Swap(DerivativeInstrument, metaclass=ABCMeta):

    def __init__(
            self,
            instrument_level_2: InstrumentLevel2,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: Union[Observable, Instrument]
    ) -> None:

        super().__init__(
            instrument_level_2=instrument_level_2,
            instrument_level_3=instrument_level_3,
            quote_currency=quote_currency,
            tradeable=False,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            underlying=underlying
        )

    @abstractmethod
    def _initialize_swap_price(self, *args) -> float:
        pass

    @abstractmethod
    def _get_swap_price(self, *args) -> float:
        pass


class SwapType(Enum):

    Payer = "Payer"
    Receiver = "Receiver"

    def __str__(self) -> str:
        return str(self.value)


class InterestRateSwap(Swap):

    def __init__(
            self,
            quote_currency: str,
            discount_curve_id: str,
            forecast_curve_id: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: InterestRate,
            payment_freq_fixed: str,
            payment_freq_float: str,
            swap_type: str,
            economy: Economy = None,
            swap_rate: float = None
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.InterestRate,
            instrument_level_3=InstrumentLevel3.InterestRateSwap,
            quote_currency=quote_currency,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            underlying=underlying
        )

        self.fixed_leg = FixedLeg(
            quote_currency=quote_currency,
            discount_curve_id=discount_curve_id,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq_fixed,
            fixed_rate=InterestRate(identifier="FIX", value=1.0, currency=self.quote_currency)  # Init used for swap rate calculation.
        )

        self.floating_leg = FloatingLeg(
            quote_currency=quote_currency,
            discount_curve_id=discount_curve_id,
            forecast_curve_id=forecast_curve_id,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq_float
        )

        self.discount_curve_id = discount_curve_id
        self.forecast_curve_id = forecast_curve_id
        self.swap_type = swap_type
        self.swap_rate = self._initialize_swap_price(economy, swap_rate)
        self.fixed_leg.fixed_rate.value = self.swap_rate

    def _initialize_swap_price(self, economy: Economy, swap_rate: float) -> float:
        if swap_rate is None:
            discount_curve = economy.yield_curves[self.discount_curve_id]
            forecast_curve = economy.yield_curves[self.forecast_curve_id]
            swap_rate = self._get_swap_price(discount_curve, forecast_curve)
        return swap_rate

    def _get_swap_price(self, discount_curve: YieldCurve, forecast_curve: YieldCurve) -> float:
        fixed_leg_value = self.fixed_leg.value(self.start_date, discount_curve)
        float_leg_value = self.floating_leg.value(self.start_date, discount_curve, forecast_curve)
        return float_leg_value / fixed_leg_value

    def value_from_economy(self, economy: Economy) -> float:
        current_date = economy.current_date
        discount_curve = economy.yield_curves[self.discount_curve_id]
        forecast_curve = economy.yield_curves[self.forecast_curve_id]
        return self.value(current_date, discount_curve, forecast_curve)

    def value(self, current_date: datetime, discount_curve: YieldCurve, forecast_curve: YieldCurve) -> float:
        fixed_leg_value = self.fixed_leg.value(current_date, discount_curve)
        float_leg_value = self.floating_leg.value(current_date, discount_curve, forecast_curve)
        # Todo: This could be more efficient.
        if self.swap_type == SwapType.Receiver.value:
            return fixed_leg_value - float_leg_value
        elif self.swap_type == SwapType.Payer.value:
            return float_leg_value - fixed_leg_value
        else:
            raise ValueError(f"Swap type {self.swap_type} not recognized!")

    def __repr__(self) -> str:
        return f"InterestRateSwap" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}"
