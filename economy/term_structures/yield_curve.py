import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import CubicSpline

from utils.dates import DateHelper, DateSchedule


class YieldCurve:

    def __init__(
            self,
            identifier: str,
            currency: str,
            tenors: np.array,
            yields: np.array,
            date_helper: DateHelper = DateHelper()
    ) -> None:

        self.identifier = identifier
        self.currency = currency
        self.tenors = tenors
        self.yields = yields
        self.date_helper = date_helper
        self.spline = self._fit_spline()
        # Todo: Obviously not the right way to go about this.
        self.old_fixing = 0.02

    def bump(self, idx: int, bump_size: float = 0.0001) -> None:
        # Todo: Allow for linear interpolation between tenor points.
        self.yields[idx] += bump_size
        self.spline = self._fit_spline()

    def _fit_spline(self) -> CubicSpline:
        return CubicSpline(self.tenors, self.yields)

    def discount_factor(self, current_date: datetime, future_date: datetime) -> float:
        tenor = self.date_helper.accrual_factor(current_date, future_date)
        return np.exp(-tenor * self.spline(tenor))

    def discount_factor_strip(self, current_date: datetime, future_dates: np.array) -> np.array:
        # Todo: Allow for additional compounding conventions.
        tenors = self.date_helper.tenors(current_date, future_dates)
        return np.exp(-tenors * self.spline(tenors))

    def forward_rate(self, current_date: datetime, accrual_start_date: datetime, accrual_end_date: datetime) -> float:
        # Todo: Method could be cleaner.
        assert current_date <= accrual_start_date
        t1 = self.date_helper.accrual_factor(current_date, accrual_start_date)
        t2 = self.date_helper.accrual_factor(current_date, accrual_end_date)
        y1, y2 = self.spline(t1), self.spline(t2)
        return (y2 * t2 - y1 * t1) / (t2 - t1)

    def forward_rate_strip(self, current_date: datetime, start_date: datetime, payment_dates: np.array) -> np.array:
        tenors = self.date_helper.tenors(current_date, payment_dates, start_date)
        yields = self.spline(tenors)
        t1, t2 = tenors[:-1], tenors[1:]
        y1, y2 = yields[:-1], yields[1:]
        fwds = (y2 * t2 - y1 * t1) / (t2 - t1)
        if len(fwds) < len(payment_dates):
            # In this case we are unable to observe the previous fixing.
            fwds = np.insert(fwds, 0, self.old_fixing)
        return fwds

    def plot(self):
        xs = np.linspace(np.min(self.tenors), np.max(self.tenors), num=100)
        plt.title(self.identifier)
        plt.plot(self.tenors, self.yields, 'o', label='data')
        plt.plot(xs, self.spline(xs), label='spline')
        plt.ylabel("Yield (%)")
        plt.xlabel("Tenor")
        plt.legend()
        plt.show()
