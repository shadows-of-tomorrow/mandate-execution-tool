from __future__ import annotations
from typing import List

from economy.base import Economy
from instruments.base import Instrument, InstrumentLevel1, InstrumentLevel2


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

    def filter_on_level_1(self, level_1s: List[InstrumentLevel1]) -> Portfolio:
        instruments = [instrument for instrument in self.instruments if instrument.instrument_level_1 in level_1s]
        return Portfolio(instruments)

    def filter_on_level_2(self, level_2s: List[InstrumentLevel2]) -> Portfolio:
        instruments = [instrument for instrument in self.instruments if instrument.instrument_level_2 in level_2s]
        return Portfolio(instruments)