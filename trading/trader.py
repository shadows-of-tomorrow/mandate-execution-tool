from trading.environment import TradingEnvironment


class Trader:

    def __init__(self, env: TradingEnvironment, policy) -> None:
        self.env = env
        self.policy = policy

    def evaluate_policy(self, render=False):
        self.eval_policy(render)

    def rollout(self, render):
        while True:
            obs, done = self.env.reset(), False
            steps, ep_ret, ep_len = 0, 0, 0
            while not done:
                steps += 1
                self._render(render)
                action = self.policy(obs).detach().numpy()
                obs, rew, done, _ = self.env.step(action)
                ep_ret += rew
            self._render(render)
            ep_len = steps
            yield ep_len, ep_ret

    def _render(self, render):
        if render:
            self.env.render()

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



