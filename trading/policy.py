import torch
import numpy as np

"""
Proximal Policy Optimization: 
-- https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""


class RolloutBuffer:

    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self) -> None:
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Policy(torch.nn.Module):

    def __init__(self, action_dim: int, state_dim: int, action_std_init: float) -> None:
        super().__init__()
        self.device = self._get_device()
        self.action_dim = action_dim
        self.action_var = None
        self.set_action_var(action_std_init)
        self.state_dim = state_dim
        self.actor = self._get_actor()
        self.critic = self._get_critic()

    def evaluate(self, state: np.array, action: np.array) -> tuple:
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        action_cov = torch.diag_embed(action_var).to(self.device)
        action_dist = torch.distributions.MultivariateNormal(action_mean, action_cov)
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = action_dist.log_prob(action)
        action_dist_entropy = action_dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, action_dist_entropy

    def act(self, state: np.array) -> tuple:
        action_mean = self.actor(state)
        action_cov = torch.diag(self.action_var).unsqueeze(dim=0)
        action_dist = torch.distributions.MultivariateNormal(action_mean, action_cov)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def _get_actor(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, self.action_dim))

    def _get_critic(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1))

    def set_action_var(self, action_std: float) -> None:
        self.action_var = torch.full((self.action_dim,), action_std ** 2).to(self.device)

    def forward(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProximalPolicyOptimization:

    def __init__(
            self,
            action_dim: int,
            state_dim: int,
            lr_actor: float,
            lr_critic: float,
            gamma: float,
            epochs: int,
            eps_clip: float,
            action_std_init: float
    ) -> None:
        self.device = self._get_device()
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.buffer = RolloutBuffer()
        self.policy = Policy(action_dim, state_dim, action_std_init).to(self.device)
        self.policy_old = Policy(action_dim, state_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = self._get_optimizer(lr_actor, lr_critic)
        self.mse_loss = torch.nn.MSELoss()

    def update(self):
        # 1. Calculate rewards.
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma + discounted_reward)
            rewards.insert(0, discounted_reward)
        # 2. Normalize rewards.
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # 3. Convert list to tensor.
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        # 4. Optimize policy for epochs.
        for _ in range(self.epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def select_action(self, state: np.array):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.detach().cpu().numpy().flatten()

    def _decay_action_std(self, action_std_decay_rate: float, action_std_min: float) -> None:
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= action_std_min:
            self.action_std = action_std_min
        self._set_action_std(self.action_std)

    def _set_action_std(self, action_std: float) -> None:
        self.action_std = action_std
        self.policy.set_action_var(action_std)
        self.policy_old.set_action_var(action_std)

    def _get_optimizer(self, lr_actor: float, lr_critic: float) -> torch.optim:
        return torch.optim.Adam([
            {"params": self.policy.actor.parameters(), 'lr': lr_actor},
            {"params": self.policy.critic.parameters(), 'lr': lr_critic}])

    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
