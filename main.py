"""
The aim of this tool is to find a "trade policy" that optimally executes
a "mandate" using a set of financial instruments. The model of the environment
will be flexible and can include stochastic scenarios or historical market data.
"""

import os
from datetime import datetime

from readers.economy_reader import EconomyReader
from readers.portfolio_reader import PortfolioReader
from readers.mandate_reader import MandateReader
from trading.environment import TradingEnvironment
from trading.trainer import PolicyTrainer

CURRENT_DATE = datetime.now()
ROOT_DIR = os.path.dirname(__file__)


def load_economy():
    economy_path = os.path.join(ROOT_DIR, 'io', 'input', 'economy')
    return EconomyReader().read_economy(CURRENT_DATE, economy_path)


def load_portfolio():
    portfolio_path = os.path.join(ROOT_DIR, 'io', 'input', 'portfolio')
    return PortfolioReader().read_portfolio(portfolio_path)


def load_mandate():
    mandate_path = os.path.join(ROOT_DIR, 'io', 'input', 'mandate')
    return MandateReader().read_mandate(mandate_path)


if __name__ == "__main__":
    portfolio, mandate, economy = load_portfolio(), load_mandate(), load_economy()
    env = TradingEnvironment(mandate=mandate, economy=economy, portfolio=portfolio)
    trainer = PolicyTrainer(env)
    trainer.train_policy()