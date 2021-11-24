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
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        # Used to reset environment.
        self.init_economy = deepcopy(economy)
        self.init_portfolio = deepcopy(portfolio)
        self.init_total_exposure_deviation = mandate.abs_exposure_deviation(portfolio, economy)
        self.init_exposure_deviations = mandate.exposure_deviations(portfolio, economy, as_array=True)
        # Trades happen in notional amounts.
        self.old_total_exposure_deviation = self.init_total_exposure_deviation
        self.new_total_exposure_deviation = None
        self.state = self.init_exposure_deviations

    def _get_observation_space(self) -> spaces:
        return spaces.Box(low=-1e30, high=1e30, shape=(self.mandate.n_exposures,))

    def _get_action_space(self) -> spaces:
        return spaces.Box(low=self.mandate.min_notional, high=self.mandate.max_notional,
                          shape=(self.mandate.n_instruments,))

    def step(self, action: np.array = None):
        if action is None:
            action = self.action_space.sample()
        assert self.action_space.contains(action)
        self._trade_instruments(action)
        reward = self._compute_reward()
        self.state = self.mandate.exposure_deviations(self.portfolio, self.economy, as_array=True)
        done = self._check_if_done()
        return self.state, reward, done, {}

    def _check_if_done(self):
        done = False
        if self.new_total_exposure_deviation < self.mandate.deviation_threshold:
            done = True
        return done

    def render(self, mode="human"):
        print(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.economy = deepcopy(self.init_economy)
        self.portfolio = deepcopy(self.init_portfolio)
        self.state = deepcopy(self.init_exposure_deviations)
        self.old_total_exposure_deviation = deepcopy(self.init_total_exposure_deviation)
        return self.state

    def _trade_instruments(self, action):
        for k in range(self.mandate.n_instruments):
            instrument = self.mandate.instrument_generators[k](action[k], self.economy)
            self.portfolio.add_instrument(instrument)

    def _compute_reward(self) -> float:
        self.new_total_exposure_deviation = self.mandate.abs_exposure_deviation(self.portfolio, self.economy)
        return self.old_total_exposure_deviation - self.new_total_exposure_deviation
