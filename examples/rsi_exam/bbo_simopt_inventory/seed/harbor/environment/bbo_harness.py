#!/usr/bin/env python3
"""Shared observable and independent scoring simulator for inventory-policy search."""
from __future__ import annotations
import hashlib
import numpy as np

def _sanitize(x,inst):
    return np.clip(np.asarray(x,dtype=np.float64),np.asarray(inst['lower'],dtype=np.float64),np.asarray(inst['upper'],dtype=np.float64))

def _replication(x,inst,rng):
    x=_sanitize(x,inst);mean=float(inst['demand_mean']);cv=float(inst['demand_cv']);periods=int(inst['periods'])
    s=.45*mean+x[0]*1.75*mean;S=s+.35*mean+x[1]*2.4*mean;trigger=x[2]*mean;smooth=.3+.7*x[3]
    on_hand=S;backlog=0.;pipeline=[];cost=0.;stockouts=0.;demand_total=0.
    for t in range(periods):
        arrivals=[q for due,q in pipeline if due<=t]
        if arrivals:on_hand+=sum(arrivals)
        pipeline=[(due,q) for due,q in pipeline if due>t]
        sigma=max(1e-6,mean*cv);shape=(mean/sigma)**2;scale=sigma*sigma/mean;demand=float(rng.gamma(shape,scale));demand_total+=demand
        served=min(on_hand,demand);on_hand-=served;lost=demand-served;backlog+=lost;stockouts+=lost
        cost+=float(inst['holding_cost'])*on_hand+float(inst['backlog_cost'])*backlog
        position=on_hand-backlog+sum(q for _,q in pipeline);target_s=smooth*s+(1.-smooth)*mean
        if position<target_s:
            qty=max(0.,S-position);lead=int(rng.poisson(float(inst['lead_mean'])))
            if backlog>trigger:lead=max(0,lead-1);cost+=float(inst['service_penalty'])*.5
            pipeline.append((t+lead+1,qty));cost+=float(inst['fixed_order_cost'])+float(inst['unit_order_cost'])*qty
        backlog*=.92
    rate=stockouts/max(demand_total,1e-9)
    return float(cost/periods+float(inst['service_penalty'])*max(0.,rate-.10)*100.)

def _observation_cost(x,inst,seed,call_count):
    base=int(inst['scenario_seed'])+7919*int(seed)+104729*int(call_count);r=np.random.default_rng(base%(2**63-1))
    return float(np.mean([_replication(x,inst,r) for _ in range(int(inst['replications']))]))

def make_objective(inst,seed=0):
    if inst.get('kind')!='stochastic_inventory_policy':raise ValueError('unsupported objective family')
    call={'n':0};lo=np.asarray(inst['lower'],float);hi=np.asarray(inst['upper'],float)
    def f(x):call['n']+=1;return _observation_cost(_sanitize(x,inst),inst,seed,call['n'])
    return f,lo,hi

def evaluate_for_scoring(inst,x):
    xx=_sanitize(x,inst);key=int(inst['scenario_seed']).to_bytes(8,'little')+xx.round(12).tobytes();base=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
    vals=[]
    for i in range(int(inst.get('scoring_replications',16))):
        seed=(base+104729*(i+1))%(2**63-1);vals.append(_replication(xx,inst,np.random.default_rng(seed)))
    return float(np.mean(vals))
