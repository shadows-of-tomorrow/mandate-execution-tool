import numpy as np
import matplotlib.pyplot as plt
from gym import Env, spaces

from mandate.base import Mandate
from economy.base import Economy
from instruments.portfolio import Portfolio


class TradingEnvironment(Env):

    def __init__(self, mandate: Mandate, economy: Economy, portfolio: Portfolio) -> None:
        super().__init__()
        self.mandate = mandate
        self.economy = economy
        self.portfolio = portfolio
        self.action_space = self._create_action_space()
        # Used to reset environment.
        self.init_economy = economy
        self.init_portfolio = portfolio
        self.init_total_exposure_deviation = mandate.abs_exposure_deviation(portfolio, economy)
        self.init_exposure_deviations = mandate.exposure_deviations(portfolio, economy, as_array=True)
        # Trades happen in notional amounts.
        self.old_total_exposure_deviation = self.init_total_exposure_deviation
        self.new_total_exposure_deviation = None
        self.max_episode_steps = 100
        self.current_episode_steps = 0
        self.state = self.init_exposure_deviations
        self.figure, self.ax = plt.subplots(figsize=(10, 10))
        self.exposure_names = [x[0].identifier for x in mandate.exposures_and_targets]
        self.ax.barh(self.exposure_names, width=self.state)
        self.figure.suptitle("Mandate Deviations")

    def _create_action_space(self) -> spaces:
        return spaces.Box(low=self.mandate.min_notional, high=self.mandate.max_notional,
                          shape=(self.mandate.n_instruments,))

    def step(self, action: np.array = None):
        if action is None:
            action = self.action_space.sample()
        assert self.action_space.contains(action)
        self._trade_instruments(action)
        reward = self._compute_reward()
        self.state = self.mandate.exposure_deviations(self.portfolio, self.economy, as_array=True)
        return self.state, reward, False, {}

    def render(self, mode="human"):
        self.ax.cla()
        self.ax.barh(self.exposure_names, width=self.state)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def reset(self):
        self.economy = self.init_economy
        self.portfolio = self.init_portfolio
        self.state = self.init_exposure_deviations
        self.old_total_exposure_deviation = self.init_total_exposure_deviation
        self.new_total_exposure_deviation = None
        self.current_episode_steps = 0
        return self.state

    def _trade_instruments(self, action):
        for k in range(self.mandate.n_instruments):
            instrument = self.mandate.instrument_generators[k](action[k], self.economy)
            self.portfolio.add_instrument(instrument)

    def _compute_reward(self) -> float:
        self.new_total_exposure_deviation = self.mandate.abs_exposure_deviation(self.portfolio, self.economy)
        return self.old_total_exposure_deviation - self.new_total_exposure_deviation
