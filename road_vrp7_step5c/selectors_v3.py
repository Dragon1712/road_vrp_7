"""UCB operator-pair selection with controlled reward denominators.

U0_gain: divide percentage gain by a GLOBAL fixed development-calibrated cost.
U1_walltime: divide by the online wall-clock duration (faithful C4 mechanism).
U2_calibrated: divide by FROZEN development-calibrated pair-specific costs.

All modes use the same UCB exploration, reward clipping, and pseudorandom path.
The global divisor in U0 normalizes scale, not pair preference.
"""
from __future__ import annotations
import math
import time
import numpy as np
from time_selector import TimeUCB

NAMES_D = ('random', 'worst', 'related', 'section')
NAMES_R = ('greedy', 'regret2')


def validate_profile(profile, n_destroy=4, n_repair=2):
    if not isinstance(profile, dict):
        raise ValueError('A frozen calibration profile is required for U0/U2.')
    p = profile.get('pair_seconds')
    required = [f'{d}|{r}' for d in NAMES_D[:n_destroy] for r in NAMES_R[:n_repair]]
    if not isinstance(p, dict) or any(x not in p for x in required):
        raise ValueError('Calibration profile missing operator pairs: ' + ','.join(x for x in required if x not in (p or {})))
    vals = [float(p[k]) for k in required]
    if any(not math.isfinite(x) or x <= 0 for x in vals):
        raise ValueError('Calibration pair costs must be strictly positive and finite.')
    scale = float(profile.get('global_seconds', 0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Calibration global_seconds must be positive and finite.')
    return vals, scale


class ControlledUCB(TimeUCB):
    def __init__(self, nd, nr, *, mode, calibration=None, c=.35, epsilon=.08):
        if mode not in ('U0_gain', 'U1_walltime', 'U2_calibrated'):
            raise ValueError('Unknown UCB reward: '+str(mode))
        super().__init__(nd, nr, c=c, epsilon=epsilon)
        if nd != 4 or nr != 2:
            raise ValueError('Controlled UCB expects exactly 4 destroys × 2 repairs')
        self.mode = mode
        self.fixed_costs, self.global_seconds = (None, None)
        if mode != 'U1_walltime':
            vals, scale = validate_profile(calibration, nd, nr)
            self.fixed_costs = np.array(vals, dtype=float).reshape(nd, nr)
            self.global_seconds = scale
        self.elapsed_sum = np.zeros((nd, nr), dtype=float)
        self.elapsed_sq_sum = np.zeros((nd, nr), dtype=float)
        self.positive_sum = np.zeros((nd, nr), dtype=float)
        self.capped_count = np.zeros((nd, nr), dtype=int)
        self.raw_reward_sum = np.zeros((nd, nr), dtype=float)

    def update(self, candidate, d_idx, r_idx, outcome):
        if self._next != (d_idx, r_idx):
            raise AssertionError('ALNS operator selection mismatch')
        elapsed = max(time.perf_counter() - self.selection_start, 1.e-6)
        pct_gain = 100. * self._candidate_gain / self._current_cost
        if self.mode == 'U1_walltime':
            denominator = elapsed
        elif self.mode == 'U0_gain':
            denominator = self.global_seconds
        else:
            denominator = float(self.fixed_costs[d_idx, r_idx])
        raw_reward = pct_gain / denominator
        reward = min(5., raw_reward)
        self.capped_count[d_idx,r_idx] += int(raw_reward >= 5.)
        self.raw_reward_sum[d_idx,r_idx] += raw_reward
        self._last_elapsed = elapsed
        self.n[d_idx, r_idx] += 1
        count = self.n[d_idx, r_idx]
        self.avg[d_idx, r_idx] += (reward - self.avg[d_idx, r_idx]) / count
        self.reward_sum += reward
        self.elapsed_sum[d_idx, r_idx] += elapsed
        self.elapsed_sq_sum[d_idx, r_idx] += elapsed * elapsed
        self.positive_sum[d_idx, r_idx] += int(self._candidate_gain > 1.e-9)

    def diagnostics(self):
        out = super().diagnostics()
        out.update(mode=self.mode,
                   global_seconds=self.global_seconds,
                   fixed_costs=None if self.fixed_costs is None else self.fixed_costs.tolist(),
                   measured_pair_elapsed_sec=self.elapsed_sum.tolist(),
                   measured_pair_elapsed_sq_sec=self.elapsed_sq_sum.tolist(),
                   positive_candidate_counts=self.positive_sum.astype(int).tolist(),
                   reward_capped_counts=self.capped_count.tolist(),
                   raw_reward_sum=self.raw_reward_sum.tolist(),
                   reward_definition='min(5, 100*positive_improvement/current_cost / time_denominator)')
        return out
