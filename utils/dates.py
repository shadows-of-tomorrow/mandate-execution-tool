from __future__ import annotations

import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DateSchedule:

    def __init__(self, start_date: datetime, payment_dates: np.array, year_fractions: np.array) -> None:
        self.start_date = start_date
        self.payment_dates = payment_dates
        self.year_fractions = year_fractions

    def clip_payment_dates(self, current_date: datetime) -> DateSchedule:
        mask = self.payment_dates > current_date
        return DateSchedule(self.start_date, self.payment_dates[mask], self.year_fractions[mask])

    def next_payment_idx(self, current_date: datetime):
        assert self.payment_dates[-1] > current_date
        mask = self.payment_dates > current_date
        return len(self.payment_dates) - len(self.payment_dates[mask])


class DateHelper:

    def __init__(self, days_in_year: int = 360, days_in_week: int = 7, days_in_month: int = 30) -> None:
        self.days_in_year = days_in_year
        self.days_in_week = days_in_week
        self.days_in_month = days_in_month

    def accrual_factor(self, start_date: datetime, end_date: datetime) -> float:
        contrib_year = self.days_in_year * (end_date.year - start_date.year)
        contrib_month = self.days_in_month * (end_date.month - start_date.month)
        contrib_day = (end_date.day - start_date.day)
        return (contrib_year + contrib_month + contrib_day) / self.days_in_year

    def tenors(self, current_date: datetime, future_dates: np.array, start_date: datetime = None):
        tenors = [self.accrual_factor(current_date, x) for x in future_dates]
        if start_date and current_date <= start_date:
            tenors.insert(0, self.accrual_factor(current_date, start_date))
        return np.array(tenors)

    def tenor_from_string(self, tenor: str) -> float:
        units, metric = int(tenor[:-1]), tenor[-1]
        if metric == "D":
            return units * (1.0/self.days_in_year)
        elif metric == "W":
            return units * (self.days_in_week/self.days_in_year)
        elif metric == "M":
            return units * (self.days_in_month/self.days_in_year)
        elif metric == "Y":
            return units


    @staticmethod
    def freq_to_delta(freq: str) -> relativedelta:
        units, metric = int(freq[:-1]), freq[-1]
        if metric == "D":
            return relativedelta(days=units)
        elif metric == "M":
            return relativedelta(months=units)
        elif metric == "Y":
            return relativedelta(years=units)
        else:
            raise ValueError(f"Time metric {metric} is not recognized!")


class DateScheduleGenerator:

    def __init__(self, payment_freq: str, date_helper: DateHelper = DateHelper()):
        self.date_helper = date_helper
        self.payment_freq = payment_freq
        self.payment_freq_delta = None
        self.date_schedule = self._get_scheduler(payment_freq)

    def _get_scheduler(self, payment_freq: str):
        if payment_freq == "Single":
            return self.date_schedule_single
        else:
            self.payment_freq_delta = self.date_helper.freq_to_delta(self.payment_freq)
            return self.date_schedule_multi

    def date_schedule_single(self, start_date: datetime, end_date: datetime) -> DateSchedule:
        accrual_factor = self.date_helper.accrual_factor(start_date, end_date)
        return DateSchedule(start_date, np.array([end_date]), np.array([accrual_factor]))

    def date_schedule_multi(self, start_date: datetime, end_date: datetime) -> DateSchedule:
        assert end_date >= start_date + self.payment_freq_delta  # Otherwise we get inconsistencies.
        payment_dates, year_fractions = [], []
        counter = 0
        while True:
            payment_date = end_date - counter * self.payment_freq_delta
            if start_date < payment_date:
                payment_dates.insert(0, payment_date)
                if counter > 0:
                    payment_date_old = payment_dates[counter - 1]
                    payment_date_new = payment_dates[counter]
                    year_fraction = self.date_helper.accrual_factor(payment_date_old, payment_date_new)
                    year_fractions.insert(0, year_fraction)
                counter += 1
            else:
                year_fraction = self.date_helper.accrual_factor(start_date, payment_dates[0])
                year_fractions.insert(0, year_fraction)
                break
        return DateSchedule(start_date, np.array(payment_dates), np.array(year_fractions))
