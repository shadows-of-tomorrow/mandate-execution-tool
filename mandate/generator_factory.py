from instruments.base import InstrumentLevel3
from instruments.factory import InstrumentFactory
from economy.base import Economy
from utils.dates import DateHelper


class InstrumentGeneratorFactory:

    def __init__(self) -> None:
        self.instrument_factory = InstrumentFactory()
        self.date_helper = DateHelper()

    def create_instrument_generator(self, instrument_level_3: str, **kwargs):
        if instrument_level_3 == InstrumentLevel3.ZeroCouponBond.value:
            return self._create_zero_coupon_bond_generator(**kwargs)
        elif instrument_level_3 == InstrumentLevel3.Stock.value:
            return self._create_stock_generator(**kwargs)
        else:
            raise ValueError(f"Instrument type {instrument_level_3} not supported!")

    def _create_zero_coupon_bond_generator(self, quote_currency: str, discount_curve_id: str, tenor: str):
        def zero_coupon_bond_generator(notional: int, economy: Economy):
            return self.instrument_factory.create_instrument(
                instrument_level_3="ZeroCouponBond",
                quote_currency=quote_currency,
                discount_curve_id=discount_curve_id,
                start_date=economy.current_date,
                maturity_date=economy.current_date + self.date_helper.freq_to_delta(tenor),
                notional=notional
            )
        return zero_coupon_bond_generator

    def _create_stock_generator(self, quote_currency: str, ticker_symbol: str):
        def stock_generator(notional: int, economy: Economy = None):
            return self.instrument_factory.create_instrument(
                instrument_level_3="Stock",
                quote_currency=quote_currency,
                notional=notional,
                ticker_symbol=ticker_symbol
            )
        return stock_generator
