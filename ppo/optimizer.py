import gym
import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from ppo.networks import ActorNN, CriticNN
from ppo.rollout import RolloutManager


class ProximalPolicyOptimization:

    def __init__(self, env, config):

        # 1. Check if environment is continuous.
        assert type(env.observation_space) == gym.spaces.Box, "Discrete observation space!"
        assert type(env.action_space) == gym.spaces.Box, "Discrete action space!"

        # 2. Initialize hyperparameters.
        self.config = config

        # 3. Initialize environment variables.
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # 4. Initialize actor network.
        self.actor = ActorNN(self.obs_dim, self.act_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.lr_actor)

        # 5. Initialize critic network.
        self.critic = CriticNN(self.obs_dim, 1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.lr_critic)

        # 6. Initialize action covariance matrix.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.config.action_std_start ** 2)
        self.cov_mat = torch.diag(self.cov_var)

        # 7. Initialize parallel rollout manager to collect data.
        self.rollout_manager = self._get_rollout_manager()

    def learn(self, total_steps):

        start_learn = time.time()
        # 1. Print start of learning process.
        self._print_start_message(total_steps)

        # 2. Initialize step counters.
        steps, iterations = 0, 0

        # 3. Repeat policy optimization for a fixed number of steps.
        while steps < total_steps:

            # 3.1 Collect data from environment by rolling out the current policy.
            start_rollout = time.time()
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.policy_rollout()
            end_rollout = time.time()

            # 3.2 Update current policy using the collected data.
            start_update = time.time()
            self.policy_update(batch_obs, batch_acts, batch_log_probs, batch_rtgs)
            end_update = time.time()

            # 3.3 Update step counters.
            steps += np.sum(batch_lens)
            iterations += 1

            # 3.4 Reduce action variance to reduce exploration.
            self._decay_action_variance(steps, total_steps)

            # 3.5 Log progress.
            avg_ep_reward = torch.sum(batch_rtgs).item() / len(batch_lens)
            print(f"* Iteration #{iterations}", flush=True)
            print(f"{'-- Steps: '}{steps}")
            print(f"{'-- Rollout + Update Time: '}{round(end_rollout-start_rollout, 2)} + {round(end_update-start_update, 2)}")
            print(f"-- Action Variance: {round(self.action_std ** 2, 3)}")
            print(f"{'-- Average Episode Reward: '}{round(avg_ep_reward, 2)}")
            print(f"\n")

        # 4. Print final messages.
        end_learn = time.time()
        print(f"* Proximal Policy Optimization Finished")
        print(f"-- Learning Time: {end_learn-start_learn}")

    def policy_update(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs):

        # 1. Calculate advantage for the last rollout.
        v, _ = self.evaluate(batch_obs, batch_acts)
        a_k = batch_rtgs - v.detach()
        a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10)

        # 2. Repeatedly loop through rollout data to update policy.
        for _ in range(self.config.updates_per_iteration):

            # 2.1 Calculate V_phi and pi_theta(a_t | s_t).
            v, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # 2.2 Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t).
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # 2.3 Calculate surrogate losses.
            surrogate_loss_1 = ratios * a_k
            surrogate_loss_2 = torch.clamp(ratios, 1 - self.config.clip, 1 + self.config.clip) * a_k

            # 2.4 Calculate actor and critic losses.
            actor_loss = (-torch.min(surrogate_loss_1, surrogate_loss_2)).mean()
            critic_loss = nn.MSELoss()(v, batch_rtgs)

            # 2.5 Calculate gradients and perform backward propagation for actor network.
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # 2.6 Calculate gradients and perform backward propagation for critic network.
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def policy_rollout(self):
        return self.rollout_manager.rollout(self.cov_mat)

    def evaluate(self, batch_obs, batch_acts):
        v = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return v, log_probs

    def _print_start_message(self, total_timesteps):
        print(f"\n")
        print(f"* Proximal Policy Optimization", flush=True)
        print(f"-- Total Steps: {total_timesteps}")
        print(f"-- Rollout Actors: {self.config.actors}")
        print(f"-- Steps Per Rollout: {self.config.steps_per_rollout}")
        print(f"-- Updates Per Iteration: {self.config.updates_per_iteration}")
        print(f"\n")

    def _decay_action_variance(self, current_timestep, total_timesteps):
        weight_timestep = current_timestep / total_timesteps
        self.action_std = self.config.action_std_start * (1.0-weight_timestep) + self.config.action_std_end * weight_timestep
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.action_std ** 2)
        self.cov_mat = torch.diag(self.cov_var)

    def _get_rollout_manager(self):
        return RolloutManager(
            env=self.env,
            actor=self.actor,
            gamma=self.config.gamma,
            steps_per_rollout=self.config.steps_per_rollout,
            steps_per_episode=self.config.steps_per_episode,
            n_workers=self.config.actors
        )