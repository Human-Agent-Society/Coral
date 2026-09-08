#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from oracle_scoring import score_minimization_traces
def main(argv):
 try:
  result=score_minimization_traces(json.loads(Path(argv[2]).read_text()),json.loads((HERE/'frozen_anchors.json').read_text()),HERE)
 except Exception as exc:result={'feasible':False,'score':0.0,'reason':f'{type(exc).__name__}: {exc}'}
 print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main(sys.argv))
