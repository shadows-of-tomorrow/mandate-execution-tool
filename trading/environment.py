import numpy as np
import matplotlib.pyplot as plt
from gym import Env, spaces
from gym.utils import seeding
from copy import deepcopy
from multiprocess import Process, Pipe

from mandate.base import Mandate
from economy.base import Economy
from instruments.portfolio import Portfolio


############################
# Trading Gym Environment #
############################
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
        # action = np.clip(action, 0.00, 0.99)
        for k in range(self.mandate.n_instruments):
            instrument = self.mandate.instrument_generators[k](action[k], self.economy)
            self.portfolio.add_instrument(instrument)

    def _get_notional(self, action, k):
        notional = ""
        for i in range(self.digits):
            notional += str(int(action[self.digits * k + i] * 10))
        return int(notional)

    def render(self, mode="human"):
        plt.clf()
        plt.barh(self.exposure_ids, width=self.state)
        plt.tight_layout()
        plt.xlim((-1.0, 1.0))
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
        return spaces.Box(low=-1e30, high=1e30, shape=(self.mandate.n_exposures, 1))

    def _get_action_space(self) -> spaces:
        return spaces.Box(low=-1e30, high=+1e30, shape=(self.mandate.n_instruments, 1))


############################
# Parallel Gym Environment #
############################
def worker(conn, env: Env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelTradingEnvironment(TradingEnvironment):

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment passed"
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.locals = []

        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return np.concatenate(results, axis=1)

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self, mode="human"):
        raise NotImplementedError
