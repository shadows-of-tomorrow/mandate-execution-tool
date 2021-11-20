import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

from utils.dates import DateHelper
from economy.base import Economy
from economy.observables.exchange_rate import ExchangeRate
from economy.observables.share_price import SharePrice
from economy.term_structures.yield_curve import YieldCurve


class EconomyFactory:

    def __init__(self) -> None:
        self.date_helper = DateHelper()
        self.yield_curve_csv = "yield_curves.csv"
        self.exchange_rate_csv = "exchange_rates.csv"
        self.share_price_csv = "share_prices.csv"

    def create_economy(self, current_date: datetime, economy_path: str) -> Economy:
        yield_curves = self._read_yield_curves(economy_path)
        exchange_rates = self._read_exchange_rates(economy_path)
        share_prices = self._read_share_prices(economy_path)
        return Economy(current_date=current_date,  yield_curves=yield_curves,
                       share_prices=share_prices, exchange_rates=exchange_rates)

    def _read_share_prices(self, economy_path: str) -> Dict[str, SharePrice]:
        shr_path = os.path.join(economy_path, self.share_price_csv)
        shr_df = pd.read_csv(shr_path)
        share_prices = shr_df.groupby(by="Identifier").apply(self._construct_share_price)
        share_prices = dict(share_prices)
        return share_prices

    def _construct_share_price(self, shr_df: pd.DataFrame) -> SharePrice:
        identifier = shr_df.iloc[0, 0]
        currency = shr_df.iloc[0, 1]
        share_price = shr_df.iloc[0, 2]
        return SharePrice(ticker_symbol=identifier, currency=currency, value=share_price)

    def _read_exchange_rates(self, economy_path: str) -> Dict[str, ExchangeRate]:
        fx_path = os.path.join(economy_path, self.exchange_rate_csv)
        fx_df = pd.read_csv(fx_path)
        exchange_rates = fx_df.groupby(by="Identifier").apply(self._construct_exchange_rate)
        exchange_rates = dict(exchange_rates)
        return exchange_rates


    def _construct_exchange_rate(self, fx_df: pd.DataFrame) -> ExchangeRate:
        base_currency = fx_df.iloc[0, 1]
        quote_currency = fx_df.iloc[0, 2]
        exchange_rate = fx_df.iloc[0, 3]
        return ExchangeRate(base_currency=base_currency, quote_currency=quote_currency, value=exchange_rate)

    def _read_yield_curves(self, economy_path: str) -> Dict[str, YieldCurve]:
        curve_path = os.path.join(economy_path, self.yield_curve_csv)
        curve_df = pd.read_csv(curve_path)
        yield_curves = curve_df.groupby(by="Identifier").apply(self._construct_yield_curve)
        yield_curves = dict(yield_curves)
        return yield_curves

    def _construct_yield_curve(self, curve_df: pd.DataFrame) -> YieldCurve:
        identifier = curve_df.iloc[0, 0]
        currency = curve_df.iloc[0, 1]
        tenors = np.array([self.date_helper.tenor_from_string(x) for x in curve_df.iloc[:, 2]])
        yields = np.array(curve_df.iloc[:, 3])
        return YieldCurve(identifier=identifier, currency=currency, tenors=tenors, yields=yields)
