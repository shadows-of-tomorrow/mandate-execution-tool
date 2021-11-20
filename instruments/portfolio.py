from typing import List

from economy.base import Economy
from instruments.base import Instrument


class Portfolio:

    def __init__(self, instruments: List[Instrument] = None) -> None:
        if instruments is None:
            self.instruments = []
        else:
            self.instruments = instruments

    def add_instrument(self, instrument: Instrument) -> None:
        self.instruments.append(instrument)

    def value(self, economy: Economy) -> float:
        return sum([instrument.value_from_economy(economy) for instrument in self.instruments])