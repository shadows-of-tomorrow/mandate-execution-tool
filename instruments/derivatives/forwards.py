from datetime import datetime
from typing import Union
from abc import ABCMeta, abstractmethod

from economy.observables.base import Observable
from economy.observables.interest_rate import InterestRate
from economy.observables.exchange_rate import ExchangeRate
from utils.dates import DateHelper
from instruments.base import Instrument, InstrumentLevel2, InstrumentLevel3
from instruments.cash.equity import Share, Stock
from economy.term_structures.yield_curve import YieldCurve
from instruments.derivatives.base import DerivativeInstrument


class Forward(DerivativeInstrument, metaclass=ABCMeta):

    def __init__(
            self,
            instrument_level_2: InstrumentLevel2,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: Union[Instrument, Observable]
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
    def _initialize_forward_price(self, *args) -> float:
        pass

    @abstractmethod
    def _get_forward_price(self, *args) -> float:
        pass


class EquityForward(Forward):

    def __init__(
            self,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: Union[Share, Stock],
            discount_curve: YieldCurve = None,
            share_price: float = None,
            forward_price: float = None
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.Equity,
            instrument_level_3=InstrumentLevel3.EquityForward,
            quote_currency=quote_currency,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            underlying=underlying
        )

        self.forward_price = self._initialize_forward_price(discount_curve, share_price, forward_price)

    def _initialize_forward_price(self, discount_curve: YieldCurve, share_price: float, forward_price: float) -> float:
        if forward_price is None:
            forward_price = self._get_forward_price(self.start_date, discount_curve, share_price)
        return forward_price

    def _get_forward_price(self, current_date: datetime, discount_curve: YieldCurve, share_price: float) -> float:
        discount_factor = discount_curve.discount_factor(current_date, self.maturity_date)
        forward_price = self.underlying.value(share_price) / discount_factor
        return forward_price

    def value(self, current_date: datetime, discount_curve: YieldCurve, share_price: float) -> float:
        assert current_date >= self.start_date
        discount_factor = discount_curve.discount_factor(current_date, self.maturity_date)
        new_forward_price = self._get_forward_price(current_date, discount_curve, share_price)
        contract_value = self.notional * (new_forward_price - self.forward_price) * discount_factor
        return contract_value

    def __repr__(self) -> str:
        return f"EquityForward" \
               f"_{repr(self.underlying)}" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}"


class ForwardRateAgreement(Forward):

    def __init__(
            self,
            quote_currency: str,
            notional: int,
            start_date: datetime,
            accrual_start_date: datetime,
            accrual_end_date: datetime,
            underlying: InterestRate,
            forecast_curve: YieldCurve = None,
            forward_price: float = None
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.InterestRate,
            instrument_level_3=InstrumentLevel3.ForwardRateAgreement,
            quote_currency=quote_currency,
            notional=notional,
            start_date=start_date,
            maturity_date=accrual_start_date,
            underlying=underlying
        )

        self.accrual_end_date = accrual_end_date
        self.accrual_factor = DateHelper().accrual_factor(accrual_start_date, accrual_end_date)
        self.forward_price = self._initialize_forward_price(forecast_curve, forward_price)

    def _initialize_forward_price(self, forecast_curve: YieldCurve, forward_price: float) -> float:
        if forward_price is None:
            forward_price = self._get_forward_price(self.start_date, forecast_curve)
        return forward_price

    def _get_forward_price(self, current_date: datetime, forecast_curve: YieldCurve) -> float:
        forward_price = forecast_curve.forward_rate(current_date, self.maturity_date, self.accrual_end_date)
        return forward_price

    def value(self, current_date: datetime, discount_curve: YieldCurve, forecast_curve: YieldCurve) -> float:
        assert current_date >= self.start_date
        new_forward_price = self._get_forward_price(current_date, forecast_curve)
        discount_factor = discount_curve.discount_factor(current_date, self.maturity_date)
        contract_value = self.notional * self.accrual_factor * (new_forward_price-self.forward_price)
        contract_value /= (1.0 + self.accrual_factor * new_forward_price)
        contract_value *= discount_factor
        return contract_value

    def __repr__(self) -> str:
        return f"ForwardRateAgreement" \
               f"_{repr(self.underlying)}" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.accrual_end_date}" \
               f"_{self.notional}"


class CurrencyForward(Forward):

    def __init__(
            self,
            quote_currency: str,
            base_currency: str,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: ExchangeRate,
            discount_curve_quote: YieldCurve = None,
            discount_curve_base: YieldCurve = None,
            spot_rate: float = None,
            forward_price: float = None
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.Currency,
            instrument_level_3=InstrumentLevel3.CurrencyForward,
            quote_currency=quote_currency,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            underlying=underlying
        )

        self.base_currency = base_currency
        self.forward_price = self._initialize_forward_price(discount_curve_quote, discount_curve_base, spot_rate, forward_price)

    def _initialize_forward_price(
            self,
            discount_curve_quote: YieldCurve,
            discount_curve_base: YieldCurve,
            spot_rate: float,
            forward_price: float
    ) -> float:

        if forward_price is None:
            forward_price = self._get_forward_price(self.start_date, discount_curve_quote, discount_curve_base, spot_rate)
        return forward_price

    def _get_forward_price(
            self,
            current_date: datetime,
            discount_curve_quote: YieldCurve,
            discount_curve_base: YieldCurve,
            spot_rate: float
    ) -> float:

        discount_factor_quote = discount_curve_quote.discount_factor(current_date, self.maturity_date)
        discount_factor_base = discount_curve_base.discount_factor(current_date, self.maturity_date)
        forward_price = spot_rate * (discount_factor_base/discount_factor_quote)
        return forward_price

    def value(
            self,
            current_date: datetime,
            discount_curve_quote: YieldCurve,
            discount_curve_base: YieldCurve,
            spot_rate: float
    ) -> float:

        assert current_date >= self.start_date
        discount_factor_quote = discount_curve_quote.discount_factor(current_date, self.maturity_date)
        new_forward_price = self._get_forward_price(current_date, discount_curve_quote, discount_curve_base, spot_rate)
        contract_value = self.notional * (new_forward_price-self.forward_price) * discount_factor_quote
        return contract_value

    def __repr__(self) -> str:
        return f"CurrencyForward" \
               f"_{repr(self.underlying)}" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}"
