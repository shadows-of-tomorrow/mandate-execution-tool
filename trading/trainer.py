import torch
import numpy as np

from trading.environment import TradingEnvironment
from trading.policy import ProximalPolicyOptimization


class TrainingConfig:

    def __init__(self) -> None:
        self.max_ep_len = 20
        self.max_training_steps = int(3e6)
        self.print_freq = self.max_ep_len * 10
        self.log_freq = self.max_ep_len * 2
        self.action_std = 0.6
        self.action_std_decay_rate = 0.05
        self.action_std_min = 0.1
        self.action_std_decay_freq = int(2.5e5)
        self.update_step = self.max_ep_len * 4
        self.epochs = 80
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.seed = 420


class PolicyTrainer:

    def __init__(self, env: TradingEnvironment, config: TrainingConfig = TrainingConfig()) -> None:
        self.env = env
        self.config = config
        self.policy_optimizer = self._get_policy_optimizer()

    def train_policy(self):
        self._set_seeds()
        self._run_training_loop()

    def _run_training_loop(self):
        step, episode = 0, 0
        while step <= self.config.max_training_steps:
            episode_reward = 0
            state = self.env.reset()
            episode += 1
            for t in range(1, self.config.max_ep_len+1):
                # 1. Select action using policy.
                action = self.policy_optimizer.select_action(state)
                state, reward, done, _ = self.env.step(action)
                # 2. Store rewards and terminals in buffer.
                self.policy_optimizer.buffer.rewards.append(reward)
                self.policy_optimizer.buffer.is_terminals.append(done)
                # 3. Update counters.
                step += 1
                episode_reward += reward
                # 4. Update policy optimizer.
                if step % self.config.update_step == 0:
                    self.policy_optimizer.update()
                # 5. Decay action std of action distribution.
                self.policy_optimizer._decay_action_std(self.config.action_std_decay_rate, self.config.action_std_min)
                # 6. Exit loop if episode done.
                if done:
                    break
            print(f"Episode: {episode}, Reward: {episode_reward}")

    def _set_seeds(self) -> None:
        torch.manual_seed(self.config.seed)
        self.env.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _get_policy_optimizer(self) -> ProximalPolicyOptimization:
        return ProximalPolicyOptimization(
            action_dim=self.env.action_space.shape[0],
            state_dim=self.env.observation_space.shape[0],
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            gamma=self.config.gamma,
            epochs=self.config.epochs,
            eps_clip=self.config.eps_clip,
            action_std_init=self.config.action_std)