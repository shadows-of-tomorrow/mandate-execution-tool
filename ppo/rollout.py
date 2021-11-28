import torch
import numpy as np
from multiprocess import Process, Pipe
from torch.distributions import MultivariateNormal


class RolloutManager:

    def __init__(self, env, actor, gamma, steps_per_rollout, steps_per_episode, n_workers):
        self.envs = self._clone_env(env, n_workers)
        self.actor = actor
        self.gamma = gamma
        self.steps_per_rollout = steps_per_rollout
        self.steps_per_episode = steps_per_episode
        self.n_workers = n_workers
        self.locals = []

        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=self.worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def rollout(self, cov_mat):
        for local in self.locals:
            local.send(("rollout", cov_mat))
        rollout_data = [self._rollout_env(self.envs[0], cov_mat)] + [local.recv() for local in self.locals]
        return self._merge_rollout_data(rollout_data)

    def worker(self, conn, env):
        while True:
            cmd, data = conn.recv()
            if cmd == "rollout":
                conn.send(self._rollout_env(env, data))

    def _rollout_env(self, env, cov_mat):
        # 1. Initialize rollout value lists.
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []

        # 2. Initialize step counter.
        steps = 0

        # 3. Collect data from multiple episodes..
        while steps < self.steps_per_rollout:

            # 3.1 Initialize episode rewards and reset environment.
            obs, ep_rews = env.reset(), []

            # 3.2 Collect data from single episode.
            for ep_t in range(self.steps_per_episode):

                # 3.2.1 Store previous state.
                batch_obs.append(obs)

                # 3.2.2 Collect data from single step.
                action, log_prob = self.get_action(obs, cov_mat)
                obs, rew, done, _ = env.step(action)

                # 3.2.3 Store data from step.
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # 3.2.4 Increment counter and check terminal condition.
                steps += 1
                if done:
                    break

            # 3.3 Store data from previous episode.
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # 4. Transform collected data to tensor format.
        batch_rtgs = self.compute_rtgs(batch_rews)

        # 5. Store rollout in buffer.
        return [batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens]

    def _merge_rollout_data(self, rollout_data):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rtgs = []
        batch_lens = []
        for k in range(len(rollout_data)):
            batch_obs += rollout_data[k][0]
            batch_acts += rollout_data[k][1]
            batch_log_probs += rollout_data[k][2]
            batch_rtgs += rollout_data[k][3]
            batch_lens += rollout_data[k][4]
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        return batch_rtgs

    def get_action(self, obs, cov_mat):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def _clone_env(self, env, n_workers):
        return [env for _ in range(n_workers)]




