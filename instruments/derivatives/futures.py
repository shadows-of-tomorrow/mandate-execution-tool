from datetime import datetime
from typing import Union
from abc import ABCMeta, abstractmethod

from economy.observables.base import Observable
from economy.observables.interest_rate import InterestRate
from instruments.base import Instrument, InstrumentLevel2, InstrumentLevel3
from instruments.cash.equity import Share, Stock
from instruments.derivatives.base import DerivativeInstrument
from economy.term_structures.yield_curve import YieldCurve


class Future(DerivativeInstrument, metaclass=ABCMeta):

    def __init__(
            self,
            instrument_level_2: InstrumentLevel2,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: Union[Instrument, Observable],
            initial_margin_rate: float,
            maintenance_margin_rate: float
    ) -> None:

        super().__init__(
            instrument_level_2=instrument_level_2,
            instrument_level_3=instrument_level_3,
            quote_currency=quote_currency,
            tradeable=True,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            underlying=underlying
        )

        self.initial_margin_rate = initial_margin_rate
        self.initial_margin = notional * initial_margin_rate
        self.maintenance_margin_rate = maintenance_margin_rate
        self.maintenance_margin = notional * initial_margin_rate * maintenance_margin_rate
        self.margin_balance = self.initial_margin

    @abstractmethod
    def _initialize_future_price(self, *args) -> float:
        pass

    @abstractmethod
    def _get_future_price(self, *args) -> float:
        pass

    @abstractmethod
    def update(self, *args) -> float:
        # Updates the future price and margin balance of the contract.
        pass

    def value(self) -> float:
        # A futures contract can be entered into at any point in time without incurring costs.
        return 0.0


class EuroDollarFuture(Future):

    def __init__(
            self,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            accrual_start_date: datetime,
            accrual_end_date: datetime,
            underlying: InterestRate,
            initial_margin_rate: float,
            maintenance_margin_rate: float,
            forecast_curve: YieldCurve = None,
            future_price: float = None
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.InterestRate,
            instrument_level_3=InstrumentLevel3.EuroDollarFuture,
            quote_currency=quote_currency,
            notional=notional,
            start_date=start_date,
            maturity_date=accrual_start_date,
            underlying=underlying,
            initial_margin_rate=initial_margin_rate,
            maintenance_margin_rate=maintenance_margin_rate
        )

        self.accrual_end_date = accrual_end_date
        self.future_price = self._initialize_future_price(forecast_curve, future_price)

    def _initialize_future_price(self, forecast_curve: YieldCurve, future_price: float) -> float:
        if future_price is None:
            future_price = self._get_future_price(self.start_date, forecast_curve)
        return future_price

    def _get_future_price(self, current_date: datetime, forecast_curve: YieldCurve) -> float:
        # Todo: Add convexity correction.
        forward_rate = forecast_curve.forward_rate(current_date, self.maturity_date, self.accrual_end_date)
        return 100.*(1.0-forward_rate)

    def update(self, current_date: datetime, forecast_curve: YieldCurve):
        new_future_price = self._get_future_price(current_date, forecast_curve)
        future_pnl = self.notional * (new_future_price-self.future_price)
        self.margin_balance += future_pnl
        self.future_price = new_future_price
        return future_pnl

    def __repr__(self) -> str:
        return f"EuroDollarFuture" \
               f"_{repr(self.underlying)}" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.accrual_end_date}" \
               f"_{self.notional}"


class EquityFuture(Future):

    def __init__(
            self,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: Union[Share, Stock],
            initial_margin_rate: float,
            maintenance_margin_rate: float,
            discount_curve: YieldCurve = None,
            share_price: float = None,
            future_price: float = None
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.Equity,
            instrument_level_3=InstrumentLevel3.EquityFuture,
            quote_currency=quote_currency,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            underlying=underlying,
            initial_margin_rate=initial_margin_rate,
            maintenance_margin_rate=maintenance_margin_rate
        )

        self.future_price = self._initialize_future_price(discount_curve, share_price, future_price)

    def _initialize_future_price(self, discount_curve: YieldCurve, share_price: float, future_price: float) -> float:
        if future_price is None:
            future_price = self._get_future_price(self.start_date, discount_curve, share_price)
        return future_price

    def _get_future_price(self, current_date: datetime, discount_curve: YieldCurve, share_price: float) -> float:
        discount_factor = discount_curve.discount_factor(current_date, self.maturity_date)
        return share_price / discount_factor

    def update(self, current_date: datetime, discount_curve: YieldCurve, share_price: float) -> float:
        new_future_price = self._get_future_price(current_date, discount_curve, share_price)
        future_pnl = self.notional * (new_future_price-self.future_price)
        self.margin_balance += future_pnl
        self.future_price = new_future_price
        return future_pnl

    def __repr__(self) -> str:
        return f"EquityFuture" \
               f"_{repr(self.underlying)}" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}"

