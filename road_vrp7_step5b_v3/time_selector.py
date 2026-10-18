"""ALNS 7.0.0 compatible cost-aware pair-selector.

Rewards are signed candidates' *positive* cost changes per measured search time,
with exploration. This is a proposed adaptation; not a literal reimplementation
of any one cited runtime-aware ALNS paper.
"""
from __future__ import annotations
import math
import time
import numpy as np


class TimeUCB:
    def __init__(self,n_destroy,n_repair,*,c=.35,epsilon=.08):
        self.nd=int(n_destroy);self.nr=int(n_repair)
        self.n=np.zeros((self.nd,self.nr),dtype=int)
        self.avg=np.zeros((self.nd,self.nr),dtype=float)
        self.c=float(c);self.epsilon=float(epsilon)
        if self.nd<1 or self.nr<1 or self.c<0 or not 0<=self.epsilon<=1:raise ValueError('Invalid UCB selector params')
        self.selection_start=0.;self._next=None;self._candidate_gain=0.;self._current_cost=1.
        self._last_elapsed=0.;self.reward_sum=0.
    def __call__(self,rng,best,curr):
        # Pull every pair at least once; otherwise cost-aware estimate never learns.
        unexplored=np.argwhere(self.n==0)
        if len(unexplored):
            idx=int(rng.integers(len(unexplored)));d,r=map(int,unexplored[idx])
        elif rng.random()<self.epsilon:
            d=int(rng.integers(self.nd));r=int(rng.integers(self.nr))
        else:
            ucb=self.avg+self.c*np.sqrt(np.log(1+self.n.sum())/self.n)
            ucb+=rng.random(ucb.shape)*1e-12
            d,r=map(int,np.unravel_index(np.argmax(ucb),ucb.shape))
        self._next=(d,r)
        self.selection_start=time.perf_counter()
        return d,r
    def record_candidate(self,before,after):
        self._candidate_gain=max(0.,float(before)-float(after))
        self._current_cost=max(float(before),1.e-6)
    def update(self,candidate,d_idx,r_idx,outcome):
        if self._next!=(d_idx,r_idx):raise AssertionError('ALNS operator selection mismatch')
        elapsed=max((time.perf_counter()-self.selection_start),1e-6)
        self._last_elapsed=elapsed
        # Relative percentage gain per SECOND, capped for robust exploration.
        reward=min(5.,100*self._candidate_gain/self._current_cost / elapsed)
        self.n[d_idx,r_idx]+=1
        count=self.n[d_idx,r_idx]
        self.avg[d_idx,r_idx]+=(reward-self.avg[d_idx,r_idx])/count
        self.reward_sum+=reward
    def diagnostics(self):return {'pair_counts':self.n.tolist(),'mean_reward':self.avg.tolist(),
                                  'total_rewards':self.reward_sum,'ucb_c':self.c,'epsilon':self.epsilon}
