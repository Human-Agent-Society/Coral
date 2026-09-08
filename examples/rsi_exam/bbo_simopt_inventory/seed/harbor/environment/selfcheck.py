#!/usr/bin/env python3
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'methods'/'main'))
from solver import Optimizer
import bbo_harness
from public_scoring import score_minimization_traces

def run(inst,seed,budget):
    objective,lo,hi=bbo_harness.make_objective(inst,seed);opt=Optimizer(len(lo),lo,hi,budget,seed,np.random.default_rng(seed));used=0;best_obs=math.inf;best_ind=math.inf;trace=[];batch=max(1,int(getattr(opt,'batch',1)))
    while used<budget:
        X=np.atleast_2d(np.asarray(opt.ask(min(batch,budget-used)),float))
        if X.ndim!=2 or X.shape[1]!=len(lo) or len(X)<1 or not np.isfinite(X).all() or np.any(X<lo) or np.any(X>hi):raise ValueError('invalid proposal batch')
        X=X[:budget-used];ys=[]
        for x in X:
            value=float(objective(x));ys.append(value);used+=1
            if value<best_obs:best_obs=value;best_ind=min(best_ind,float(bbo_harness.evaluate_for_scoring(inst,x)))
            trace.append(best_ind)
        try:opt.tell(X,np.asarray(ys,float),None)
        except TypeError:opt.tell(X,np.asarray(ys,float))
        batch=max(1,int(getattr(opt,'batch',batch)))
    return trace

def main():
    data=json.loads((ROOT/'data'/'visible.json').read_text());anchors=json.loads((ROOT/'data'/'visible_anchors.json').read_text());traces=[]
    for i,inst in enumerate(data['instances']):traces.append([run(inst,0x9E3779B1*(i+1)+j,int(data['budget'])) for j in range(int(anchors['n_seeds']))])
    result=score_minimization_traces({'traces':traces},anchors,ROOT/'data');print(f"visible raw_metric = {result['raw_metric']:.12f}; mapped reward = {result['score']:.12f}; {len(data['instances'])} cases x {anchors['n_seeds']} repeats x {data['budget']} queries")
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__=='__main__':main()
