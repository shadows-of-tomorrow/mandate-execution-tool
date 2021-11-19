import numpy as np
from datetime import datetime
from abc import ABCMeta, abstractmethod

from instruments.base import InstrumentLevel2, InstrumentLevel3
from instruments.cash.base import CashInstrument
from economy.observables.interest_rate import InterestRate
from utils.dates import DateScheduleGenerator, DateSchedule
from utils.cash_flows import CashFlowSchedule
from economy.term_structures.yield_curve import YieldCurve


class Loan(CashInstrument, metaclass=ABCMeta):

    def __init__(
            self,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            tradeable: bool,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str
    ) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.Debt,
            instrument_level_3=instrument_level_3,
            quote_currency=quote_currency,
            tradeable=tradeable
        )

        self.notional = notional
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.payment_freq = payment_freq
        self.schedule_generator = DateScheduleGenerator(payment_freq)
        self.date_schedule = self.schedule_generator.date_schedule(start_date, maturity_date)

    @abstractmethod
    def _generate_cash_flows(self, *args) -> CashFlowSchedule:
        pass

    def _clip_payment_dates(self, current_date: datetime) -> DateSchedule:
        return self.date_schedule.clip_payment_dates(current_date)


class FixedRateLoan(Loan):

    def __init__(
            self,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            tradeable: bool,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str,
            fixed_rate: InterestRate
    ) -> None:

        super().__init__(
            instrument_level_3=instrument_level_3,
            quote_currency=quote_currency,
            tradeable=tradeable,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq
        )

        self.fixed_rate = fixed_rate

    def value(self, current_date: datetime, discount_curve: YieldCurve) -> float:
        cash_flows = self._generate_cash_flows(current_date)
        return cash_flows.present_value(current_date, discount_curve)

    def _generate_cash_flows(self, current_date: datetime) -> CashFlowSchedule:
        schedule = self._clip_payment_dates(current_date)
        cash_flows = self.notional * schedule.year_fractions * self.fixed_rate.value
        if len(cash_flows) > 0:
            cash_flows[-1] += self.notional
        return CashFlowSchedule(schedule.payment_dates, cash_flows)

    def __repr__(self):
        return f"FixedRateLoan" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}" \
               f"_{self.payment_freq}" \
               f"_{self.fixed_rate.value}"


class FloatingRateLoan(Loan):

    def __init__(
            self,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            tradeable: bool,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str
    ) -> None:

        super().__init__(
            instrument_level_3=instrument_level_3,
            quote_currency=quote_currency,
            tradeable=tradeable,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq
        )

    def value(self, current_date: datetime, discount_curve: YieldCurve, forecast_curve: YieldCurve) -> float:
        cash_flows = self._generate_cash_flows(current_date, forecast_curve)
        return cash_flows.present_value(current_date, discount_curve)

    def _generate_cash_flows(self, current_date: datetime, forecast_curve: YieldCurve) -> CashFlowSchedule:
        schedule = self._clip_payment_dates(current_date)
        forward_rates = forecast_curve.forward_rate_strip(current_date, self.start_date, schedule.payment_dates)
        cash_flows = self.notional * schedule.year_fractions * forward_rates
        cash_flows[-1] += self.notional
        return CashFlowSchedule(schedule.payment_dates, cash_flows)

    def __repr__(self):
        return f"FloatingRateLoan" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}" \
               f"_{self.payment_freq}"


class ZeroCouponBond(Loan):

    def __init__(
            self,
            quote_currency: str,
            notional: float,
            start_date: datetime,
            maturity_date: datetime
    ) -> None:

        super().__init__(
            instrument_level_3=InstrumentLevel3.ZeroCouponBond,
            quote_currency=quote_currency,
            tradeable=True,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq="Single"
        )

    def value(self, current_date: datetime, discount_curve: YieldCurve) -> float:
        cash_flows = self._generate_cash_flows(current_date)
        return cash_flows.present_value(current_date, discount_curve)

    def _generate_cash_flows(self, current_date: datetime) -> CashFlowSchedule:
        schedule = self._clip_payment_dates(current_date)
        if len(schedule.payment_dates) == 0:
            cash_flows = np.array([])
        elif len(schedule.payment_dates) == 1:
            cash_flows = np.array([self.notional])
        else:
            raise ValueError(f"Zero coupon bond can not have more than one payment.")
        return CashFlowSchedule(schedule.payment_dates, cash_flows)

    def __repr__(self) -> str:
        return f"ZeroCouponBond" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}"


class FixedRateBond(FixedRateLoan):

    def __init__(
            self,
            quote_currency: str,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str,
            fixed_rate: InterestRate
    ) -> None:

        super().__init__(
            instrument_level_3=InstrumentLevel3.FixedRateBond,
            quote_currency=quote_currency,
            tradeable=True,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq,
            fixed_rate=fixed_rate
        )

    def _accrued_interest(self, current_date: datetime) -> float:
        # Todo: Integrate with bond valuation.
        accrued_interest = 0.0
        if current_date < self.maturity_date:
            next_payment_idx = self.date_schedule.next_payment_idx(current_date)
            if next_payment_idx == 0:
                accrual_start_date = self.date_schedule.start_date
            else:
                accrual_start_date = self.date_schedule.payment_dates[next_payment_idx-1]
            accrual_period_current = self.schedule_generator.date_helper.accrual_factor(accrual_start_date, current_date)
            accrued_interest = self.notional * self.fixed_rate.value * accrual_period_current
        return accrued_interest

    def clean_price(self, current_date: datetime, discount_curve: YieldCurve) -> float:
        dirty_price = self.value(current_date, discount_curve)
        accrued_interest = self._accrued_interest(current_date)
        return dirty_price - accrued_interest

    def __repr__(self) -> str:
        return f"FixedRateBond" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}" \
               f"_{self.payment_freq}" \
               f"_{self.fixed_rate.value}"


class FloatingRateBond(FloatingRateLoan):

    def __init__(
            self,
            quote_currency: str,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str
    ) -> None:

        super().__init__(
            instrument_level_3=InstrumentLevel3.FloatingRateBond,
            quote_currency=quote_currency,
            tradeable=True,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq
        )

    def __repr__(self) -> str:
        return f"FloatingRateBond" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}" \
               f"_{self.payment_freq}"


class FixedLeg(FixedRateLoan):

    def __init__(
            self,
            quote_currency: str,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str,
            fixed_rate: InterestRate
    ) -> None:

        super().__init__(
            instrument_level_3=InstrumentLevel3.FixedLeg,
            quote_currency=quote_currency,
            tradeable=False,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq,
            fixed_rate=fixed_rate
        )

    def _generate_cash_flows(self, current_date: datetime) -> CashFlowSchedule:
        schedule = self._clip_payment_dates(current_date)
        cash_flows = self.notional * schedule.year_fractions * self.fixed_rate.value
        return CashFlowSchedule(schedule.payment_dates, cash_flows)

    def __repr__(self) -> str:
        return f"FixedLeg" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}" \
               f"_{self.payment_freq}"


class FloatingLeg(FloatingRateLoan):

    def __init__(
            self,
            quote_currency: str,
            notional: float,
            start_date: datetime,
            maturity_date: datetime,
            payment_freq: str
    ) -> None:

        super().__init__(
            instrument_level_3=InstrumentLevel3.FloatingLeg,
            quote_currency=quote_currency,
            tradeable=False,
            notional=notional,
            start_date=start_date,
            maturity_date=maturity_date,
            payment_freq=payment_freq
        )

    def _generate_cash_flows(self, current_date: datetime, forecast_curve: YieldCurve) -> CashFlowSchedule:
        schedule = self._clip_payment_dates(current_date)
        forward_rates = forecast_curve.forward_rate_strip(current_date, self.start_date, schedule.payment_dates)
        cash_flows = self.notional * schedule.year_fractions * forward_rates
        return CashFlowSchedule(schedule.payment_dates, cash_flows)

    def __repr__(self) -> str:
        return f"FloatingLeg" \
               f"_{self.start_date}" \
               f"_{self.maturity_date}" \
               f"_{self.notional}" \
               f"_{self.payment_freq}"
