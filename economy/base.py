from typing import Dict
from datetime import datetime

from economy.observables.exchange_rate import ExchangeRate
from economy.observables.share_price import SharePrice
from economy.term_structures.yield_curve import YieldCurve


class Economy:

    def __init__(
            self,
            current_date: datetime,
            yield_curves: Dict[str, YieldCurve],
            share_prices: Dict[str, SharePrice],
            exchange_rates: Dict[str, ExchangeRate]
    ) -> None:

        self.current_date = current_date
        self.yield_curves = yield_curves
        self.share_prices = share_prices
        self.exchange_rates = exchange_rates

