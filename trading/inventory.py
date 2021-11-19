from typing import List

from instruments.base import Instrument


class Inventory:

    def __init__(self, instruments: List[Instrument] = None) -> None:
        if instruments is None:
            self.instruments = []
        else:
            self.instruments = instruments

    def add_instrument(self, instrument: Instrument) -> None:
        self.instruments.append(instrument)

    def value(self, *args, **kwargs) -> float:
        return sum([instrument.value(*args, **kwargs) for instrument in self.instruments])