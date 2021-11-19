"""
This module contains the fundamental building blocks of derivative instruments.
"""

from abc import ABCMeta
from datetime import datetime
from typing import Union

from economy.observables.base import Observable
from instruments.base import Instrument, InstrumentLevel1, InstrumentLevel2, InstrumentLevel3


class DerivativeInstrument(Instrument, metaclass=ABCMeta):
    """
    A derivative instrument is a financial instrument which derives
    its value from the value and characteristics of one or more underlining entities.
    """

    def __init__(
            self,
            instrument_level_2: InstrumentLevel2,
            instrument_level_3: InstrumentLevel3,
            quote_currency: str,
            tradeable: bool,
            notional: int,
            start_date: datetime,
            maturity_date: datetime,
            underlying: Union[Instrument, Observable]
    ) -> None:

        super().__init__(
            instrument_level_1=InstrumentLevel1.Derivative,
            instrument_level_2=instrument_level_2,
            instrument_level_3=instrument_level_3,
            quote_currency=quote_currency,
            tradeable=tradeable
        )

        self.notional = notional
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.underlying = underlying
