from instruments.base import InstrumentLevel2, InstrumentLevel3
from instruments.cash.base import CashInstrument


class Share(CashInstrument):

    def __init__(self, quote_currency: str, ticker_symbol: str) -> None:
        self.ticker_symbol = ticker_symbol
        super().__init__(
            instrument_level_2=InstrumentLevel2.Equity,
            instrument_level_3=InstrumentLevel3.Share,
            quote_currency=quote_currency,
            tradeable=True
        )

    def value(self, share_price: float) -> float:
        return share_price

    def __repr__(self) -> str:
        return f"Share_{self.ticker_symbol}"


class Stock(CashInstrument):

    def __init__(self, quote_currency: str, ticker_symbol: str, number_of_shares: int) -> None:
        assert number_of_shares > 1
        self.share = Share(quote_currency=quote_currency, ticker_symbol=ticker_symbol)
        self.number_of_shares = number_of_shares
        super().__init__(
            instrument_level_2=InstrumentLevel2.Equity,
            instrument_level_3=InstrumentLevel3.Stock,
            quote_currency=quote_currency, tradeable=True
        )

    def value(self, share_price: float) -> float:
        return self.share.value(share_price) * self.number_of_shares

    def __repr__(self) -> str:
        return f"Stock_{self.share.ticker_symbol}_{self.number_of_shares}"
