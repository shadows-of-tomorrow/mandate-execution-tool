import torch

from ppo.model import PPO
from ppo.network import FeedForwardNN
from trading.environment import TradingEnvironment

HYPERPARAMETERS = {
    'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 1,
    'gamma': 0.90,
    'n_updates_per_iteration': 10,
    'lr': 3e-4,
    'clip': 0.2,
    'render': False,
    'render_every_i': 10
}


class TrainingConfig:

    def __init__(self) -> None:
        self.total_timesteps = 200_000
        self.hyperparameters = HYPERPARAMETERS


class PolicyTrainer:

    def __init__(self, env: TradingEnvironment, config: TrainingConfig = TrainingConfig()) -> None:
        self.env = env
        self.config = config
        self.model = self._get_ppo_model()

    def learn_policy(self) -> FeedForwardNN:
        self.model.learn(total_timesteps=self.config.total_timesteps)
        return self._get_policy()

    def _get_policy(self) -> FeedForwardNN:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        policy = FeedForwardNN(state_dim, action_dim)
        policy.load_state_dict(self.model.actor.state_dict())
        return policy

    def _get_ppo_model(self):
        return PPO(policy_class=FeedForwardNN, env=self.env, **self.config.hyperparameters)
