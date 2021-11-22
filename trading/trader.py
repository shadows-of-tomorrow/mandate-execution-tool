from mandate.base import Mandate
from instruments.portfolio import Portfolio
from economy.base import Economy


class Trader:

    def __init__(self, mandate: Mandate, portfolio: Portfolio) -> None:
        self.mandate = mandate
        self.portfolio = portfolio

    def check_mandate(self, economy: Economy) -> dict:
        return self.mandate.exposure_deviation(self.portfolio, economy)
