"""
The aim of this tool is to find a trading policy that optimally executes
a mandate using a set of financial instruments. The model of the environment
will be flexible and can include stochastic scenarios or historical market data.
"""

import os
from datetime import datetime

from readers.economy_reader import EconomyReader
from readers.portfolio_reader import PortfolioReader
from readers.mandate_reader import MandateReader
from trading.environment import TradingEnvironment
from trading.trader import Trader
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


def get_environment():
    from instruments.portfolio import Portfolio
    portfolio, mandate, economy = Portfolio(), load_mandate(), load_economy()
    return TradingEnvironment(mandate=mandate, economy=economy, portfolio=portfolio)


if __name__ == "__main__":
    env = get_environment()
    trade_policy = PolicyTrainer(env).learn_policy()
    Trader(env, trade_policy).evaluate_policy(render=True)
