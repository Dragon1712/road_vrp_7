"""Store *same-trajectory* historical best solutions, never interpolate future bests.

Timestamp is the perf_counter offset from search invocation.  Recording only new
bests has small overhead; road ex-post replay happens AFTER ALNS search ends.
Snapshots past the actually completed search are explicitly CENSORED.
"""
from __future__ import annotations
import math
import time


def frozen(routes):
    return tuple(tuple(int(x) for x in r) for r in routes)


class BestTrace:
    def __init__(self, initial_routes, initial_cost):
        if not math.isfinite(float(initial_cost)):
            raise ValueError('Invalid initial objective')
        self.initial = frozen(initial_routes)
        self.initial_cost = float(initial_cost)
        self.events = [(0., self.initial_cost, self.initial)]
        self.start = None

    def begin(self):
        if self.start is not None:
            raise ValueError('Trace already started')
        self.start = time.perf_counter()

    def note_best(self, cost, routes):
        if self.start is None:
            raise ValueError('Trace not started')
        elapsed = time.perf_counter() - self.start
        cost = float(cost)
        if cost < self.events[-1][1] - 1.e-7:
            self.events.append((elapsed, cost, frozen(routes)))

    def snapshots(self, thresholds, search_elapsed):
        if self.start is None:
            raise ValueError('Trace not started')
        out = []
        for sec in thresholds:
            sec = float(sec)
            if not math.isfinite(sec) or sec <= 0:
                raise ValueError('Snapshot times must be positive and finite')
            reached = search_elapsed + 1.e-8 >= sec
            # Even when the run ends early, an unreached threshold never counts
            # as a budget observation: no hypothetical future value is supplied.
            if reached:
                events = [event for event in self.events if event[0] <= sec + 1.e-9]
                event = events[-1]
                out.append({'target_sec':sec,'status':'reached',
                            'best_elapsed_sec':event[0], 'planned_cost':event[1],
                            'routes':[list(x) for x in event[2]]})
            else:
                out.append({'target_sec':sec,'status':'censored',
                            'best_elapsed_sec':None,'planned_cost':None,'routes':None})
        return out


def parse_times(spec):
    v = tuple(float(t.strip()) for t in spec.split(',') if t.strip())
    if not v or len(v) != len(set(v)) or v != tuple(sorted(v)) or any(not math.isfinite(s) or s <= 0 for s in v):
        raise ValueError('Snapshots must be ascending unique positive seconds')
    return v
