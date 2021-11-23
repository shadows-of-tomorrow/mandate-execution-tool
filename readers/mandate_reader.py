import os
import pandas as pd

from mandate.base import Mandate
from exposures.factory import ExposureFactory
from mandate.generator_factory import InstrumentGeneratorFactory


class MandateReader:

    def __init__(self) -> None:
        self.exposure_csv = "exposures.csv"
        self.instrument_csv = "instruments.csv"
        self.exposure_factory = ExposureFactory()
        self.instrument_generator_factory = InstrumentGeneratorFactory()

    def read_mandate(self, mandate_path: str) -> Mandate:
        exposures_and_targets = self._read_exposures_and_targets(mandate_path)
        instrument_generators = self._read_instrument_generators(mandate_path)
        return Mandate(exposures_and_targets, instrument_generators)

    def _read_exposures_and_targets(self, mandate_path: str) -> list:
        # Todo: Ugly reader but it works for now...
        exposures_and_targets = []
        exposure_path = os.path.join(mandate_path, self.exposure_csv)
        exposure_df = pd.read_csv(exposure_path)
        for k in range(len(exposure_df)):
            kwargs = {'identifier': exposure_df.iloc[k, 0]}
            if not pd.isnull(exposure_df.iloc[k, 2]):
                kwargs['curve_identifier'] = exposure_df.iloc[k, 2]
            if not pd.isnull(exposure_df.iloc[k, 3]):
                kwargs['tenor'] = exposure_df.iloc[k, 3]
            exposure_type = exposure_df.iloc[k, 1]
            target = exposure_df.iloc[k, 4]
            exposure = self.exposure_factory.create_exposure(exposure_type, **kwargs)
            exposures_and_targets.append((exposure, target))
        return exposures_and_targets

    def _read_instrument_generators(self, mandate_path: str):
        instrument_generators = []
        instrument_path = os.path.join(mandate_path, self.instrument_csv)
        instrument_df = pd.read_csv(instrument_path)
        for k in range(len(instrument_df)):
            identifier = instrument_df.iloc[k, 0]
            instrument_type = instrument_df.iloc[k, 1]
            quote_currency = instrument_df.iloc[k, 2]
            discount_curve_id = instrument_df.iloc[k, 3]
            tenor = instrument_df.iloc[k, 4]
            ticker_symbol = instrument_df.iloc[k, 5]
            kwargs = {'quote_currency': quote_currency}
            if not pd.isnull(discount_curve_id):
                kwargs['discount_curve_id'] = discount_curve_id
            if not pd.isnull(tenor):
                kwargs['tenor'] = tenor
            if not pd.isnull(ticker_symbol):
                kwargs['ticker_symbol'] = ticker_symbol
            instrument_generator = self.instrument_generator_factory.create_instrument_generator(instrument_type, **kwargs)
            instrument_generators.append(instrument_generator)
        return instrument_generators