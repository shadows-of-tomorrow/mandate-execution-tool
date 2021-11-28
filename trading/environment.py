import numpy as np
import matplotlib.pyplot as plt
from gym import Env, spaces
from gym.utils import seeding
from copy import deepcopy

from mandate.base import Mandate
from economy.base import Economy
from instruments.portfolio import Portfolio


class TradingEnvironment(Env):

    def __init__(self, mandate: Mandate, economy: Economy, portfolio: Portfolio) -> None:
        super().__init__()
        self.mandate = mandate
        self.economy = economy
        self.portfolio = portfolio
        self.digits = 2
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        # Used to reset environment.
        self.init_economy = deepcopy(economy)
        self.init_portfolio = deepcopy(portfolio)
        self.init_exposure_deviations = mandate.exposure_deviations(portfolio, economy, as_array=True)
        # Trades happen in notional amounts.
        self.old_state = deepcopy(self.init_exposure_deviations)
        self.state = deepcopy(self.init_exposure_deviations)
        self.exposure_ids = [x.identifier for x in self.mandate.exposures]
        self.steps = 0
        self.max_steps = 10

    def step(self, action: np.array) -> tuple:
        self.steps += 1
        self._trade_instruments(action)
        self._update_state()
        reward, done = self._compute_reward()
        self.old_state = deepcopy(self.state)
        return self.state, reward, done, {}

    def _compute_reward(self):
        reward, done = np.sum(np.abs(self.old_state) - np.abs(self.state)), False
        if self.steps >= self.max_steps:
            done = True
        return reward, done

    def _trade_instruments(self, action: np.array) -> None:
        for k in range(self.mandate.n_instruments):
            instrument = self.mandate.instrument_generators[k](action[k], self.economy)
            self.portfolio.add_instrument(instrument)

    def render(self, mode="human"):
        plt.clf()
        plt.title("Mandate Exposures")
        exposures, targets = self.mandate.portfolio_exposures(self.portfolio, self.economy), self.mandate.targets
        plt.grid()
        plt.barh(self.exposure_ids, width=exposures)
        plt.barh(self.exposure_ids, width=targets, alpha=0.50)
        plt.tight_layout()
        plt.pause(0.00001)

    def _update_state(self) -> None:
        self.state = self.mandate.exposure_deviations(self.portfolio, self.economy, as_array=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.economy = deepcopy(self.init_economy)
        self.portfolio = deepcopy(self.init_portfolio)
        self.state = deepcopy(self.init_exposure_deviations)
        self.old_state = deepcopy(self.state)
        self.steps = 0
        return self.state

    def _get_observation_space(self) -> spaces:
        return spaces.Box(low=-1.0, high=+1.0, shape=(self.mandate.n_exposures,))

    def _get_action_space(self) -> spaces:
        return spaces.Box(low=-1.0, high=+1.0, shape=(self.mandate.n_instruments,))
