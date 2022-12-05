import json
from collections import defaultdict
from datetime import timedelta
from timeit import default_timer


class Timing:
    def __init__(self) -> None:
        self.times = defaultdict(list)
        self.running_timers = dict()

    def start(self, timer: str):
        if timer in self.running_timers:
            raise Exception(f"Timer {timer} already running")

        self.running_timers[timer] = default_timer()

    def stop(self, timer: str | None = None, all: bool = False):
        assert timer or all

        if not all and timer not in self.running_timers:
            raise Exception(f"Timer {timer} not running")

        if all:
            for timer, start in self.running_timers.items():
                self.times[timer].append(default_timer() - start)
            self.running_timers = dict()
        else:
            self.times[timer].append(default_timer() - self.running_timers[timer])
            del self.running_timers[timer]

    def write(self, f):
        times_ = {k: (v if len(v) > 1 else v[0]) for k, v in self.times.items()}
        json.dump(times_, f, indent=2)

    def summary(self, aggregate=True):
        if aggregate:
            time_strs = [
                str(timedelta(seconds=sum(time_list))) for time_list in self.times
            ]
        else:
            time_strs = [
                "({}) {}".format(
                    timedelta(seconds=sum(time_list)),
                    ", ".join([f"{time:.2s}" for time in time_list]),
                )
                for time_list in self.times
            ]

        for timer, time_str in zip(self.times.keys(), time_strs):
            print("{}: {}".format(timer, time_str))
