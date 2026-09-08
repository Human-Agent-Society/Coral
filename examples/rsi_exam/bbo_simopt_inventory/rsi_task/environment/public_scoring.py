"""Piecewise reference/upper scoring for frozen inventory-policy traces."""
from __future__ import annotations
import json,math
from pathlib import Path
from typing import Any
import numpy as np
SCORING_NAME='mean_log_cost_progress_auc70_final30';ANYTIME_WEIGHT=.70;FINAL_WEIGHT=.30

def _coerce(decision,anchors):
    t=np.asarray(decision.get('traces'),float);shape=(int(anchors.get('n_hidden',anchors.get('n_instances'))),int(anchors['n_seeds']),int(anchors['budget']))
    if t.shape!=shape or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t,axis=2)>1e-8):raise ValueError('invalid best-so-far trace tensor')
    return t

def _progress(values,baseline):
    den=np.log1p(baseline)
    if np.any(den<=1e-12):raise ValueError('baseline traces must be positive')
    return np.clip((den-np.log1p(values))/den,0.,1.)

def _combine(p):return ANYTIME_WEIGHT*np.mean(p[:,:-1],axis=1)+FINAL_WEIGHT*p[:,-1]
def _shape(f):
    f=np.clip(f,0.,1.);return 1.-np.log1p(3.*(1.-f))/math.log(4.)
def _map(q,ref):
    if np.any(ref<=1e-9) or np.any(ref>=1.):raise ValueError('reference qualities must lie strictly between baseline and upper')
    out=np.zeros_like(q);lo=(q>0)&(q<=ref);hi=q>ref;out[lo]=.3*_shape(q[lo]/ref[lo]);out[hi]=.3+.7*_shape((q[hi]-ref[hi])/(1.-ref[hi]));return np.clip(out,0.,1.)

def score_minimization_traces(decision,anchors,here):
    t=_coerce(decision,anchors);med=np.median(t,axis=1);floor=np.asarray(anchors['floor_trace_median'],float);ref=np.asarray(anchors['ref_trace_median'],float);shape=(int(anchors.get('n_hidden',anchors.get('n_instances'))),int(anchors['budget']))
    if floor.shape!=shape or ref.shape!=shape or np.any(np.diff(floor,axis=1)>1e-8) or np.any(np.diff(ref,axis=1)>1e-8):raise ValueError('invalid frozen anchor traces')
    p=_progress(med,floor);rp=_progress(ref,floor);raw=_combine(p);rraw=_combine(rp);reward_i=_map(raw,rraw)
    anytime=np.mean(p[:,:-1],axis=1);final=p[:,-1]
    return {'feasible':True,'metric':SCORING_NAME,'score':float(np.mean(reward_i)),'raw_metric':float(np.mean(raw)),'reference_raw_metric':float(np.mean(rraw)),'score_anytime_raw':float(np.mean(anytime)),'score_final_raw':float(np.mean(final)),'kpi_final_cost_median':float(np.median(med[:,-1])),'reward_per_inst':reward_i.tolist(),'raw_quality_per_inst':raw.tolist(),'reference_quality_per_inst':rraw.tolist(),'anytime_raw_per_inst':anytime.tolist(),'final_raw_per_inst':final.tolist(),'normalization_kind':'per-instance log1p cost progress with piecewise reference and theoretical zero-cost upper','theoretical_raw_cost_lower_bound':0.0,'reason':None}
