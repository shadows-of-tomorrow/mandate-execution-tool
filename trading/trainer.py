from ppo.optimizer import ProximalPolicyOptimization
from ppo.networks import ActorNN
from trading.environment import TradingEnvironment


class TrainingConfig:

    def __init__(self) -> None:
        self.total_timesteps = 1_000_000
        self.actors = 4
        self.steps_per_rollout = 600
        self.steps_per_episode = 1000
        self.updates_per_iteration = 40
        self.gamma = 0.99
        self.lr_actor = 0.0003
        self.lr_critic = 0.0003
        self.action_std_start = 0.60
        self.action_std_end = 0.10
        self.clip = 0.2


class PolicyTrainer:

    def __init__(self, env: TradingEnvironment, config: TrainingConfig = TrainingConfig()) -> None:
        self.env = env
        self.config = config
        self.model = self._get_ppo_model()

    def learn_policy(self) -> ActorNN:
        self.model.learn(total_steps=self.config.total_timesteps)
        return self._get_policy()

    def _get_policy(self) -> ActorNN:
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        policy = ActorNN(state_dim, action_dim)
        policy.load_state_dict(self.model.actor.state_dict())
        return policy

    def _get_ppo_model(self):
        return ProximalPolicyOptimization(env=self.env, config=self.config)
