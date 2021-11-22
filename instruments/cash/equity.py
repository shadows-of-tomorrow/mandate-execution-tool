from instruments.base import InstrumentLevel2, InstrumentLevel3
from instruments.cash.base import CashInstrument
from economy.base import Economy


class Share(CashInstrument):

    def __init__(self, quote_currency: str, ticker_symbol: str) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.Equity,
            instrument_level_3=InstrumentLevel3.Share,
            quote_currency=quote_currency,
            tradeable=True
        )

        self.ticker_symbol = ticker_symbol

    def value_from_economy(self, economy: Economy) -> float:
        share_price = economy.share_prices[self.ticker_symbol].value
        return self.value(share_price)

    def value(self, share_price: float) -> float:
        return share_price


class Stock(CashInstrument):

    def __init__(self, quote_currency: str, ticker_symbol: str, number_of_shares: int) -> None:

        super().__init__(
            instrument_level_2=InstrumentLevel2.Equity,
            instrument_level_3=InstrumentLevel3.Stock,
            quote_currency=quote_currency, tradeable=True
        )

        self.share = Share(quote_currency=quote_currency, ticker_symbol=ticker_symbol)
        self.number_of_shares = number_of_shares

    def value_from_economy(self, economy: Economy) -> float:
        share_price = economy.share_prices[self.share.ticker_symbol].value
        return self.value(share_price)

    def value(self, share_price: float) -> float:
        return self.share.value(share_price) * self.number_of_shares
