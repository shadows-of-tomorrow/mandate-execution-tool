import os
import pandas as pd

from mandate.base import Mandate
from exposures.factory import ExposureFactory


class MandateReader:

    def __init__(self) -> None:
        self.exposure_csv = "exposures.csv"
        self.exposure_factory = ExposureFactory()

    def read_mandate(self, mandate_path: str) -> Mandate:
        exposures_and_targets = self._read_exposures_and_targets(mandate_path)
        return Mandate(exposures_and_targets)

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
