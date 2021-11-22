"""
Temporary burner test file.
"""
from datetime import datetime

from readers.economy_reader import EconomyReader
from readers.portfolio_reader import PortfolioReader
from readers.mandate_reader import MandateReader
from trading.trader import Trader

CURRENT_DATE = datetime.now()


def load_economy():
    economy_path = "C:/Users/robin/Desktop/Repos/mandate-execution-tool/io/input/economy/"
    return EconomyReader().read_economy(CURRENT_DATE, economy_path)


def load_portfolio():
    portfolio_path = "C:/Users/robin/Desktop/Repos/mandate-execution-tool/io/input/portfolio/"
    return PortfolioReader().read_portfolio(portfolio_path)


def load_mandate():
    mandate_path = "C:/Users/robin/Desktop/Repos/mandate-execution-tool/io/input/mandate/"
    return MandateReader().read_mandate(mandate_path)


if __name__ == "__main__":
    portfolio, mandate = load_portfolio(), load_mandate()
    trader = Trader(mandate=mandate, portfolio=portfolio)
    economy = load_economy()
    mandate = trader.check_mandate(economy)
    print(mandate)