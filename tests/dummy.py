"""
Temporary burner test file.
"""
from datetime import datetime

from readers.economy_reader import EconomyReader
from readers.portfolio_reader import PortfolioReader
from readers.mandate_reader import MandateReader
from trading.trader import Trader
from trading.environment import TradingEnvironment

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
    portfolio, mandate, economy = load_portfolio(), load_mandate(), load_economy()
    env = TradingEnvironment(mandate=mandate, economy=economy, portfolio=portfolio)
    trader = Trader(env)