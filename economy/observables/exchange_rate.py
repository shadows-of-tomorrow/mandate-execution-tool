from economy.observables.base import Observable


class ExchangeRate(Observable):
    """
    An exchange rate is the rate at which we can exchange a single unit
    of the base currency for the quote currency.
    """

    def __init__(self, value: float, base_currency: str, quote_currency: str):
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        symbol = f"{self.base_currency}_{self.quote_currency}"
        super().__init__(identifier=symbol, value=value)

    def flip(self) -> None:
        base_placeholder = self.base_currency
        self.base_currency = self.quote_currency
        self.quote_currency = base_placeholder
        self.value = 1.0 / self.value
        self.identifier = f"{self.base_currency}_{self.quote_currency}"
