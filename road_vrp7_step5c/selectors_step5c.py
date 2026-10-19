"""Official ALNS bandit adapters and a frozen joint-pair LRW-inspired policy.

C9 is explicitly an independent adaptation inspired by GRLOS LRW; it is NOT
an exact reproduction of the original GRLOS implementation.
"""
from __future__ import annotations
import math
import time
import numpy as np

D_NAMES = ('random', 'worst', 'related', 'section')
R_NAMES = ('greedy', 'regret2')
PAIRS = tuple(f'{d}|{r}' for d in D_NAMES for r in R_NAMES)


class OfficialBanditAdapter:
    def __init__(self, algorithm, seed, nd=4, nr=2):
        if nd != 4 or nr != 2:
            raise ValueError('Expected 4 destroys and 2 repairs')
        from alns.select import AlphaUCB
        if algorithm == 'C7_alpha_ucb':
            # Same four categorical outcome scores across official bandit baselines.
            self.inner = AlphaUCB(scores=[8.,4.,1.,0.],alpha=.05,
                                  num_destroy=nd,num_repair=nr)
        elif algorithm == 'C8_thompson':
            try:
                from alns.select import MABSelector
                from mabwiser.mab import LearningPolicy
            except ImportError as e:
                raise RuntimeError('Install optional dependencies: pip install "alns[mabwiser]==7.0.0"') from e
            self.inner = MABSelector(scores=[8.,4.,1.,0.],num_destroy=nd,
                num_repair=nr,learning_policy=LearningPolicy.ThompsonSampling(binarizer=lambda arm, reward: int(reward > 0)),seed=int(seed))
        else:
            raise ValueError(algorithm)
        self.name=algorithm
        self.pair_counts=np.zeros((nd,nr),dtype=int)
        self.elapsed_sum=np.zeros((nd,nr),dtype=float)
        self._start=0.
    def __call__(self,rng,best,current):
        d,r=self.inner(rng,best,current)
        self._start=time.perf_counter()
        return int(d),int(r)
    def record_candidate(self,before,after):
        pass # ALNS built-in selectors update from official outcome, not continuous improvement.
    def update(self,candidate,d_idx,r_idx,outcome):
        self.inner.update(candidate,d_idx,r_idx,outcome)
        self.pair_counts[d_idx,r_idx]+=1
        self.elapsed_sum[d_idx,r_idx]+=max(0.,time.perf_counter()-self._start)
    def diagnostics(self):
        return {'type':self.name,'pair_counts':self.pair_counts.tolist(),
                'mean_reward':np.zeros_like(self.pair_counts,dtype=float).tolist(),
                'measured_pair_elapsed_sec':self.elapsed_sum.tolist(),
                'reward_definition':'official ALNS categorical outcomes [8,4,1,0]'}


class FrozenPairLRW:
    """A fixed pair-weight roulette with offline-learned weights; no online update."""
    def __init__(self,profile,nd=4,nr=2):
        if nd!=4 or nr!=2:raise ValueError('Expected 4x2 operator pairs')
        if not isinstance(profile,dict) or profile.get('schema')!='lrw_pair_weights_v1':
            raise ValueError('Invalid LRW training profile')
        weights=profile.get('weights',{})
        if set(weights)!=set(PAIRS):raise ValueError('LRW profile must define all eight pairs')
        p=np.array([float(weights[k]) for k in PAIRS],dtype=float)
        if not np.isfinite(p).all() or not (p>0).all():raise ValueError('Invalid LRW weights')
        self.probabilities=p/p.sum()
        self.pair_counts=np.zeros((nd,nr),dtype=int)
        self._start=0.
        self.elapsed_sum=np.zeros((nd,nr),dtype=float)
    def __call__(self,rng,best,current):
        idx=int(rng.choice(len(PAIRS),p=self.probabilities))
        self._start=time.perf_counter()
        return idx//2,idx%2
    def record_candidate(self,before,after):pass
    def update(self,candidate,d_idx,r_idx,outcome):
        self.pair_counts[d_idx,r_idx]+=1
        self.elapsed_sum[d_idx,r_idx]+=max(0.,time.perf_counter()-self._start)
    def diagnostics(self):
        return {'type':'C9_lrw_frozen_pair_weights','pair_counts':self.pair_counts.tolist(),
                'mean_reward':np.zeros((4,2)).tolist(),
                'measured_pair_elapsed_sec':self.elapsed_sum.tolist(),
                'frozen_pair_probabilities':self.probabilities.reshape(4,2).tolist(),
                'reward_definition':'offline learned from development; no online updates'}


def make_step5c_selector(name,seed,profile=None,nd=4,nr=2):
    if name in ('C7_alpha_ucb','C8_thompson'):
        return OfficialBanditAdapter(name,seed,nd,nr)
    if name=='C9_lrw':return FrozenPairLRW(profile,nd,nr)
    raise ValueError(name)
