"""
This module contains the fundamental building blocks of cash instruments.
"""

from abc import ABCMeta
from instruments.base import Instrument, InstrumentLevel1, InstrumentLevel2, InstrumentLevel3


class CashInstrument(Instrument, metaclass=ABCMeta):
    """
    A cash instrument is a financial instrument whose value is determined directly by market observables.
    """
    def __init__(
            self,
            instrument_level_2: InstrumentLevel2,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            tradeable: bool
    ) -> None:

        super().__init__(
            quote_currency=quote_currency,
            instrument_level_1=InstrumentLevel1.Cash,
            instrument_level_2=instrument_level_2,
            instrument_level_3=instrument_level_3,
            tradeable=tradeable
        )