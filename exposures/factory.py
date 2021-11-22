from exposures.base import Exposure, ExposureType
from exposures.base import AssetAllocationDebt, AssetAllocationEquity
from exposures.base import ZeroDelta


class ExposureFactory:

    def create_exposure(self, exposure_type: str, **kwargs) -> Exposure:
        if exposure_type == ExposureType.AssetAllocationDebt.value:
            return self._create_asset_allocation_debt(**kwargs)
        elif exposure_type == ExposureType.AssetAllocationEquity.value:
            return self._create_asset_allocation_equity(**kwargs)
        elif exposure_type == ExposureType.ZeroDelta.value:
            return self._create_zero_delta(**kwargs)

    @staticmethod
    def _create_asset_allocation_debt(identifier: str) -> AssetAllocationDebt:
        return AssetAllocationDebt(identifier=identifier)

    @staticmethod
    def _create_asset_allocation_equity(identifier: str) -> AssetAllocationEquity:
        return AssetAllocationEquity(identifier=identifier)

    @staticmethod
    def _create_zero_delta(identifier: str, curve_identifier: str, tenor: float) -> ZeroDelta:
        return ZeroDelta(identifier=identifier, curve_identifier=curve_identifier, tenor=tenor)
