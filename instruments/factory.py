from datetime import datetime
from typing import Union

from economy.observables.interest_rate import InterestRate
from economy.observables.exchange_rate import ExchangeRate
from economy.term_structures.yield_curve import YieldCurve
from instruments.base import Instrument, InstrumentLevel3
from instruments.cash.equity import Share, Stock
from instruments.cash.debt import ZeroCouponBond, FixedRateBond, FloatingRateBond
from instruments.derivatives.forwards import EquityForward, ForwardRateAgreement, CurrencyForward
from instruments.derivatives.futures import EquityFuture, EuroDollarFuture
from instruments.derivatives.swaps import InterestRateSwap


class InstrumentFactory:

    def create_instrument(self, instrument_level_3: str, **kwargs) -> Instrument:
        if instrument_level_3 == InstrumentLevel3.Share.value:
            return self._create_share(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.Stock.value:
            return self._create_stock(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.ZeroCouponBond.value:
            return self._create_zero_coupon_bond(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.FixedRateBond.value:
            return self._create_fixed_rate_bond(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.FloatingRateBond.value:
            return self._create_floating_rate_bond(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.EquityForward.value:
            return self._create_equity_forward(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.ForwardRateAgreement.value:
            return self._create_forward_rate_agreement(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.CurrencyForward.value:
            return self._create_currency_forward(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.EquityFuture.value:
            return self._create_equity_future(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.EuroDollarFuture.value:
            return self._create_eurodollar_future(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.InterestRateSwap.value:
            return self._create_interest_rate_swap(**kwargs)
        else:
            raise ValueError(f"Instrument <{instrument_level_3}> not recognized!")

    @staticmethod
    def _create_share(quote_currency: str, ticker_symbol: str) -> Share:
        return Share(quote_currency=quote_currency, ticker_symbol=ticker_symbol)

    @staticmethod
    def _create_stock(quote_currency: str, ticker_symbol: str, number_of_shares: int) -> Stock:
        return Stock(quote_currency=quote_currency, ticker_symbol=ticker_symbol, number_of_shares=number_of_shares)

    @staticmethod
    def _create_zero_coupon_bond(quote_currency: str, notional: int, start_date: datetime,
                                 maturity_date: datetime) -> ZeroCouponBond:
        return ZeroCouponBond(quote_currency=quote_currency, notional=notional, start_date=start_date,
                              maturity_date=maturity_date)

    @staticmethod
    def _create_fixed_rate_bond(quote_currency: str, notional: int, start_date: datetime,
                                maturity_date: datetime, payment_freq: str, fixed_rate: float) -> FixedRateBond:
        return FixedRateBond(quote_currency=quote_currency, notional=notional, start_date=start_date,
                             maturity_date=maturity_date, payment_freq=payment_freq,
                             fixed_rate=InterestRate(identifier="FIXED_RATE", value=fixed_rate))

    @staticmethod
    def _create_floating_rate_bond(quote_currency: str, notional: int, start_date: datetime,
                                   maturity_date: datetime, payment_freq: str) -> FloatingRateBond:
        return FloatingRateBond(quote_currency=quote_currency, notional=notional, start_date=start_date,
                                maturity_date=maturity_date, payment_freq=payment_freq)

    @staticmethod
    def _create_equity_forward(quote_currency: str, notional: int, start_date: datetime,
                               maturity_date: datetime, underlying: Union[Share, Stock],
                               discount_curve: YieldCurve = None, share_price: float = None,
                               forward_price: float = None) -> EquityForward:
        return EquityForward(quote_currency=quote_currency, notional=notional, start_date=start_date,
                             maturity_date=maturity_date, underlying=underlying, discount_curve=discount_curve,
                             forward_price=forward_price, share_price=share_price)

    @staticmethod
    def _create_forward_rate_agreement(quote_currency: str, notional: int, start_date: datetime,
                                       accrual_start_date: datetime, accrual_end_date: datetime,
                                       underlying: InterestRate, forecast_curve: YieldCurve = None,
                                       forward_price: float = None) -> ForwardRateAgreement:
        return ForwardRateAgreement(quote_currency=quote_currency, notional=notional, start_date=start_date,
                                    accrual_start_date=accrual_start_date, accrual_end_date=accrual_end_date,
                                    underlying=underlying, forecast_curve=forecast_curve, forward_price=forward_price)

    @staticmethod
    def _create_currency_forward(quote_currency: str, base_currency: str, notional: int,
                                 start_date: datetime, maturity_date: datetime, underlying: ExchangeRate,
                                 discount_curve_quote: YieldCurve = None, discount_curve_base: YieldCurve = None,
                                 spot_rate: float = None, forward_price: float = None) -> CurrencyForward:
        return CurrencyForward(quote_currency=quote_currency, base_currency=base_currency, notional=notional,
                               start_date=start_date, maturity_date=maturity_date, underlying=underlying,
                               discount_curve_quote=discount_curve_quote, discount_curve_base=discount_curve_base,
                               spot_rate=spot_rate, forward_price=forward_price)

    @staticmethod
    def _create_equity_future(quote_currency: str, notional: int, start_date: datetime, maturity_date: datetime,
                              underlying: Union[Share, Stock], initial_margin_rate: float, maintenance_margin_rate: float,
                              discount_curve: YieldCurve = None, share_price: float = None,
                              future_price: float = None) -> EquityFuture:
        return EquityFuture(quote_currency=quote_currency, notional=notional, start_date=start_date,
                            maturity_date=maturity_date, underlying=underlying, initial_margin_rate=initial_margin_rate,
                            maintenance_margin_rate=maintenance_margin_rate, discount_curve=discount_curve,
                            share_price=share_price, future_price=future_price)

    @staticmethod
    def _create_eurodollar_future(quote_currency: str, notional: int, start_date: datetime, accrual_start_date: datetime,
                                  accrual_end_date: datetime, underlying: InterestRate, initial_margin_rate: float,
                                  maintenance_margin_rate: float, forecast_curve: YieldCurve = None,
                                  future_price: float = None) -> EuroDollarFuture:
        return EuroDollarFuture(quote_currency=quote_currency, notional=notional, start_date=start_date,
                                accrual_start_date=accrual_start_date, accrual_end_date=accrual_end_date,
                                underlying=underlying, initial_margin_rate=initial_margin_rate,
                                maintenance_margin_rate=maintenance_margin_rate, forecast_curve=forecast_curve,
                                future_price=future_price)

    @staticmethod
    def _create_interest_rate_swap(quote_currency: str, notional: int, start_date: datetime, maturity_date: datetime,
                                   underlying: InterestRate, payment_freq_fixed: str, payment_freq_float: str,
                                   swap_type: str, discount_curve: YieldCurve = None, forecast_curve: YieldCurve = None,
                                   swap_rate: float = None) -> InterestRateSwap:
        return InterestRateSwap(quote_currency=quote_currency, notional=notional, start_date=start_date,
                                maturity_date=maturity_date, underlying=underlying,
                                payment_freq_fixed=payment_freq_fixed, payment_freq_float=payment_freq_float,
                                swap_type=swap_type, discount_curve=discount_curve, forecast_curve=forecast_curve,
                                swap_rate=swap_rate)
