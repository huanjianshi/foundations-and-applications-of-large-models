import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from advanced_retrieval import BM25,tokenize,weighted_rrf,late_fusion_order,minmax,KeywordEngine

class AdvancedChecks(unittest.TestCase):
 def test_bm25_exact_and_unknown(self):
  b=BM25([{'title':'梯度下降','text':'调整参数'},{'title':'低秩适配','text':'冻结权重'}])
  self.assertGreater(b.scores('梯度下降')[0],b.scores('梯度下降')[1]);self.assertEqual(b.scores('zzzz').sum(),0)
  self.assertIn('lora',tokenize('LoRA低秩适配'))
 def test_zero_lexical_cannot_vote(self):
  self.assertEqual(weighted_rrf(np.zeros(3),np.array([.2,.9,.1]),['a','b','c']),[1,0,2])
 def test_candidate_boundary(self):
  o=late_fusion_order([0,1,2],np.array([.1,.9,.3]),np.array([.2,.3,.1]),np.array([.5,.9,100]),['a','b','c'],2,[0,0,1])
  self.assertEqual(o,[1,0,2])
 def test_degenerate_normalization(self):
  np.testing.assert_equal(minmax([2,2,2]),[0,0,0]);np.testing.assert_equal(minmax([1,3]),[0,1])
 def test_cpu_only_engine(self):
  e=KeywordEngine([{'id':'a','title':'梯度下降','text':'调整参数','source':'source'}])
  self.assertEqual(e.search('梯度',k=1)['results'][0]['id'],'a')
  with self.assertRaises(ValueError):e.search('x','optimized')

if __name__=='__main__':unittest.main()
