from trading.environment import TradingEnvironment


class Trader:

    def __init__(self, env: TradingEnvironment) -> None:
        self.env = env
