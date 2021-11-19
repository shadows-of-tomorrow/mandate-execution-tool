import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from economy.term_structures.yield_curve import YieldCurve


class CashFlowSchedule:

    def __init__(self, payment_dates: np.array, cash_flows: np.array) -> None:
        assert len(payment_dates) == len(cash_flows)
        self.payment_dates = payment_dates
        self.cash_flows = cash_flows

    def present_value(self, current_date: datetime, discount_curve: YieldCurve) -> float:
        discount_factors = discount_curve.discount_factor_strip(current_date, self.payment_dates)
        return np.inner(self.cash_flows, discount_factors)

    def plot(self):
        # 1. Get masks for positive / negative bars.
        mask_positive = self.cash_flows >= 0
        mask_negative = self.cash_flows < 0
        # 2. Plot cash flows against payment date.
        plt.title("Cash Flow Schedule")
        plt.xlabel("Payment Date")
        plt.ylabel("Cash Flow")
        plt.bar(self.payment_dates[mask_positive], self.cash_flows[mask_positive], color="green")
        plt.bar(self.payment_dates[mask_negative], self.cash_flows[mask_negative], color="red")
        plt.grid()
        plt.tight_layout()
        plt.show()
