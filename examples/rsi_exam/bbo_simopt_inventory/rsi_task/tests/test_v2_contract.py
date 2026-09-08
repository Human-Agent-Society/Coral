from __future__ import annotations
import importlib.util,json,re,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class V2Contract(unittest.TestCase):
 def test_harbor_shape_and_anonymous_instruction(self):
  for p in ['instruction.md','task.toml','environment/Dockerfile','solution/solve.sh','tests/Dockerfile','tests/test.sh','tests/grade.py']:self.assertTrue((ROOT/p).is_file(),p)
  text=(ROOT/'instruction.md').read_text();self.assertEqual(len(re.findall(r'^# ',text,re.M)),1);self.assertEqual(len(re.findall(r'^## ',text,re.M)),5)
  self.assertNotRegex(text.lower(),r'reward\s*(?:0\.3|0\.6|1(?:\.0)?)|quadratic|surrogate|reference_optimizer|hidden[_ -]?seed')
 def test_public_anchor_landings(self):
  a=json.loads((ROOT/'environment/data/visible_anchors.json').read_text());sc=mod(ROOT/'environment/public_scoring.py','sc')
  for field,expected in [('floor_trace_median',0.0),('ref_trace_median',0.3)]:
   t=np.asarray(a[field],float)[:,None,:];t=np.repeat(t,int(a['n_seeds']),axis=1);out=sc.score_minimization_traces({'traces':t},a,ROOT/'environment/data');self.assertAlmostEqual(out['score'],expected,places=12)
  z=np.zeros((a['n_instances'],a['n_seeds'],a['budget']));out=sc.score_minimization_traces({'traces':z.tolist()},a,ROOT/'environment/data');self.assertAlmostEqual(out['score'],1.0,places=12)
 def test_powered_confirmation_and_nonnegative_cost_proof(self):
  result=json.loads((ROOT/'_dev/rebuild_v2/powered_confirmation_result.json').read_text());self.assertEqual(result['status'],'PASS');self.assertTrue(all(result['gates'].values()))
  for rel in ['environment/data/visible.json','_dev/rebuild_v2/powered_confirmation.json']:
   for x in json.loads((ROOT/rel).read_text())['instances']:
    for k in ['holding_cost','backlog_cost','fixed_order_cost','unit_order_cost','service_penalty']:self.assertGreaterEqual(x[k],0)
 def test_shared_simulator_bytes(self):self.assertEqual((ROOT/'environment/bbo_harness.py').read_bytes(),(ROOT/'tests/heldout/bbo_harness.py').read_bytes())
if __name__=='__main__':unittest.main()
