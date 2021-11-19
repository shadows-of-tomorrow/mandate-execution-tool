
class Observable:
    """
    Something which has a (numerical) value that can be observed in the marketplace.
    While this class may seem redundant it is conceptually important for two reasons.

    (1.) In principle, derivatives can be based on values ascribed to virtually anything.
         To encapsulate the range of possible "anythings" this class is a useful tool.

    (2.) Many intermediary classes such as term structures are derived (bootstrapped) from
         observations in the market. Often we would like to observe the impact on valuations
         from a change in market behavior instead of "engineered" pricing constructs.
    """
    def __init__(self, identifier: str, value: float) -> None:
        self.identifier = identifier
        self.value = value

    def __repr__(self) -> str:
        return self.identifier
