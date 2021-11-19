import inspect

from economy.base import Economy
from instruments.base import Instrument


class InstrumentPricer:

    def __init__(self) -> None:
        pass

    def _extract_arguments(self, instrument: Instrument, economy: Economy):
        pass
