import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from retrieval import Lexical,metrics,rank,fuse,Engine

class Checks(unittest.TestCase):
    def test_multi_relevant(self):
        r=metrics(['b','a','c','d','e'],['a','d'])
        self.assertEqual(r['recall@1'],0);self.assertEqual(r['recall@3'],.5)
        self.assertEqual(r['recall@5'],1);self.assertEqual(r['mrr@5'],.5)
        self.assertEqual(metrics(['a'],['b'])['mrr@5'],0)
    def test_rrf_hand(self):
        v=fuse([[0,1,2],[1,2,0]],3,60)
        self.assertAlmostEqual(v[0],1/61+1/63)
        self.assertEqual(rank(v,['a','b','c']),[1,0,2])
    def test_tfidf_hand(self):
        x=Lexical(['甲乙甲乙','甲乙丙丁'])
        self.assertAlmostEqual(x.idf[x.pos['甲乙']],1)
        self.assertAlmostEqual(x.idf[x.pos['乙甲']],np.log(3/2)+1)
        scores,zero=x.scores('甲乙甲乙')
        self.assertFalse(zero);self.assertAlmostEqual(scores[0],1)
        self.assertTrue(x.scores('zzz')[1])
    def test_tie_and_duplicates(self):
        self.assertEqual(rank([0,0],['b','a']),[1,0])
        with self.assertRaises(ValueError): metrics(['a','a'],['a'])
    def test_search_validation(self):
        e=object.__new__(Engine);e.cards=[{'id':'a'}]
        for q,m,k in [('', 'hybrid',1),('x','bad',1),('x','lexical',0)]:
            with self.assertRaises(ValueError): e.search(q,m,k)

if __name__=='__main__': unittest.main()
