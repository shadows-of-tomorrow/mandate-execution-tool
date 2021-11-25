from ppo.model import PPO
from ppo.network import ActorNN
from trading.environment import TradingEnvironment, ParallelTradingEnvironment

HYPERPARAMETERS = {
    'timesteps_per_batch': 2048,
    'max_timesteps_per_episode': 10000,
    'gamma': 0.95,
    'n_updates_per_iteration': 10,
    'lr_actor': 0.0003,
    'lr_critic': 0.0003,
    'action_std_start': 10.00,
    'action_std_end': 0.10,
    'clip': 0.2,
    'render': False,
    'render_every_i': 10
}


class TrainingConfig:

    def __init__(self) -> None:
        self.workers = 1  # Todo: Fix parallel computing.
        self.total_timesteps = 2_000_000
        self.hyperparameters = HYPERPARAMETERS


class PolicyTrainer:

    def __init__(self, env: TradingEnvironment, config: TrainingConfig = TrainingConfig()) -> None:
        self.config = config
        self.env = self._construct_env(env)
        self.model = self._get_ppo_model()

    def learn_policy(self) -> ActorNN:
        self.model.learn(total_timesteps=self.config.total_timesteps)
        return self._get_policy()

    def _get_policy(self) -> ActorNN:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        policy = ActorNN(state_dim, action_dim)
        policy.load_state_dict(self.model.actor.state_dict())
        return policy

    def _get_ppo_model(self):
        return PPO(env=self.env, **self.config.hyperparameters)

    def _construct_env(self, env: TradingEnvironment):
        if self.config.workers > 1:
            envs = [env for _ in range(self.config.workers)]
            env = ParallelTradingEnvironment(envs)
        return env
