"""Step6-C1-R2: exact frozen C3 shadow draw and paired Section features.

The shadow draw copies the RNG and selector statistics and NEVER changes the
original search RNG. Its action equals the next real C3 action if called from
an unchanged RNG state in the same frozen search state.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np


def state_key(task_id, scenario_id, routes):
    # MUST match frozen benchmark_30s_r.state_identifier byte-for-byte.
    return hashlib.sha256(json.dumps([str(task_id),str(scenario_id),
                  tuple(tuple(int(c) for c in r) for r in routes)],sort_keys=True).encode()).hexdigest()[:24]


def clone_rng(rng):
    if not isinstance(rng,np.random.Generator):
        raise TypeError('R2 only supports numpy.random.Generator')
    copied=np.random.Generator(type(rng.bit_generator)())
    copied.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    return copied


def block_action(partial):
    blocks=getattr(partial,'blocks',())
    if not blocks:return None
    if len(blocks)!=1:raise ValueError('C3 must choose one Section')
    b=blocks[0]
    return {'route':int(b.route),'pos':int(b.pos),'size':len(b.ids),
            'block_ids':list(map(int,b.ids))}


def matches(a,b):
    return a is not None and b is not None and all(int(a[k])==int(b[k]) for k in ('route','pos','size'))


def shadow_c3(selector, mode, routes, rng, frozen_destroy, stats_factory, return_snapshot=False):
    """Use the *actual* C3 algorithm with an RNG clone and isolated stats.

    The cloned Selector shares only immutable obj/support configuration; the
    frozen C3 method reads candidates and writes only its own stats in v3.
    """
    shadow=copy.copy(selector)
    shadow.stats=stats_factory()
    rngcopy=clone_rng(rng)
    partial=frozen_destroy(shadow,mode,routes,rngcopy)
    if return_snapshot:
        return (block_action(partial),partial,copy.deepcopy(rngcopy.bit_generator.state),
                dict(shadow.stats.d))
    return block_action(partial)


def transfer_shadow(rng,selector,after_state,stat_delta,include_stats):
    """Commit cloned C3 RNG draws. On defer also commit C3 selector statistics.

    In learning override, C3 draws are still committed so later ALNS random
    numbers are coupled to original C3 at this choice point.
    """
    rng.bit_generator.state=copy.deepcopy(after_state)
    if include_stats:
        for key,v in stat_delta.items():
            if v:selector.stats.add(key,v)


def paired_features(routes, action, obj, hist, hour, weekday, mode='F2', cost=None):
    from features import FEATURE_MODES,core_online,extract_boundary
    f=core_online(routes,action,obj,hour,weekday,cost)
    extra=extract_boundary(routes,action,obj,hist if mode=='F4' else None)
    if extra is None:return None
    f.update(extra)
    values=np.array([f[x] for x in FEATURE_MODES[mode]],np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite features')
    return values


def action_identity(a):
    if a is None:return None
    return (int(a['route']),int(a['pos']),int(a['size']))
