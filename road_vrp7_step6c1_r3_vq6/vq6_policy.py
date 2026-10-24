"""Frozen F2 online selectors compatible with Step6-C1-R2 real-C3 shadow hook.

Model features are 3 * 26 = 78, from candidate / real-C3 / difference.
Road.replay, repair outcomes, and future traffic are NEVER predictor inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from r2_policy import R2Advantage,paired_matrix

KINDS=('E0_MATCHED','RESID_OOF50','DEFER_RESID')

@dataclass
class VQ6Policy(R2Advantage):
    kind: str = 'E0_MATCHED'
    residual_model: object = None
    alpha: float = 0.

    def predict_delta(self,actions,baseline):
        if self.kind not in KINDS or self.mode!='F2' or self.model is None:
            raise ValueError('Invalid VQ6 policy state')
        xx=np.asarray(actions,np.float32);b=np.asarray(baseline,np.float32)
        if xx.ndim!=2 or xx.shape[1]!=26 or b.shape!=(26,):
            raise ValueError('Online schema requires 26 F2 candidate and C3 features')
        features=paired_matrix(xx,b)
        values=np.asarray(self.model.predict(features),float)
        if self.kind in ('RESID_OOF50','DEFER_RESID'):
            if self.residual_model is None or self.alpha!=.5:
                raise ValueError('Missing frozen 0.5 OOF residual predictor')
            values=values+.5*np.asarray(self.residual_model.predict(features),float)
        if values.shape!=(len(xx),) or not np.isfinite(values).all():
            raise ValueError('Nonfinite or mis-shaped predictions')
        return values

    def select(self,actions,baseline,same=None,valid=None):
        if self.kind=='DEFER_RESID':
            # Identical model scoring cost before guaranteed real-C3 fallback.
            scores=self.predict_delta(actions,baseline)
            return None,float(np.max(scores)) if len(scores) else float('-inf'),'forced_defer_after_inference'
        return super().select(actions,baseline,same,valid)
