from economy.observables.base import Observable


class SharePrice(Observable):

    def __init__(self, ticker_symbol: str, currency: str, value: float) -> None:
        self.currency = currency
        super().__init__(identifier=ticker_symbol, value=value)