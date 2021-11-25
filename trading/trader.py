from trading.environment import TradingEnvironment


class Trader:

    def __init__(self, env: TradingEnvironment, policy) -> None:
        self.env = env
        self.policy = policy

    def evaluate_policy(self, render=False):
        self.eval_policy(render)

    def rollout(self, render):
        # Rollout until user kills process
        while True:
            obs = self.env.reset()
            done = False

            # number of timesteps so far
            t = 0

            # Logging data
            ep_len = 0  # episodic length
            ep_ret = 0  # episodic return

            while not done:
                t += 1

                # Render environment if specified, off by default
                if render:
                    self.env.render()

                # Query deterministic action from policy and run it
                action = self.policy(obs).detach().numpy()
                obs, rew, done, _ = self.env.step(action)

                # Sum all episodic rewards as we go along
                ep_ret += rew

            if render:
                self.env.render()

            # Track episodic length
            ep_len = t

            # returns episodic length and return in this iteration
            yield ep_len, ep_ret

    def _log_summary(self, ep_len, ep_ret, ep_num):
        ep_len = str(round(ep_len, 2))
        ep_ret = str(round(ep_ret, 2))
        print(flush=True)
        print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
        print(f"Episodic Length: {ep_len}", flush=True)
        print(f"Episodic Return: {ep_ret}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

    def eval_policy(self, render=False):
        # Rollout with the policy and environment, and log each episode's data
        for ep_num, (ep_len, ep_ret) in enumerate(self.rollout(render)):
            self._log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)



