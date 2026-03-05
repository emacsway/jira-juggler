import datetime

import businesstimedelta

__all__ = ('AddWorkingDays',)


class AddWorkingDays:
    def __init__(self, weeklymax):
        self._workday = businesstimedelta.WorkDayRule(
            start_time=datetime.time(9),
            end_time=datetime.time(18),
            working_days=list(range(weeklymax)))

        # Take out the lunch break
        self._lunch_break = businesstimedelta.LunchTimeRule(
            start_time=datetime.time(12),
            end_time=datetime.time(13),
            working_days=list(range(weeklymax)))

        # Combine the two
        self._business_hrs = businesstimedelta.Rules([self._workday, self._lunch_break])

    def __call__(self, from_date, add_days):
        delta = businesstimedelta.BusinessTimeDelta(self._business_hrs, timedelta=datetime.timedelta(days=add_days))
        return from_date + delta
