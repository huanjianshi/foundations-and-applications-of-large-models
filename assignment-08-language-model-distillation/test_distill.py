"""Checks for the loss contract, causal model, evaluation state and paired runs."""
import unittest
import torch
from torch.nn import functional as F
import distill_lm as d


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tok = d.Tokenizer(d.build_corpus())
        d.TOKENIZER = cls.tok

    def test_corpus_and_parameter_counts(self):
        self.assertEqual((len(d.build_corpus()),len(d.build_corpus(300,999)),self.tok.vocab_size),(49457,5357,322))
        for c,n in [(160,737280),(64,147456)]:
            m = d.GPT(322,d.config(c))
            self.assertEqual(sum(p.numel() for p in m.parameters()),n)
            approximate = 2*322*c+96*c+24*c*c
            # Include 2 blocks' biases/LNs and the final LN for an exact hand count.
            self.assertEqual(approximate+28*c,n)

    def test_token_averaging_and_exact_gradient(self):
        torch.manual_seed(4)
        s = torch.randn(2,3,7,dtype=torch.float64,requires_grad=True)
        t = torch.randn_like(s,requires_grad=True)
        temperature = 4.
        kd = d.kd_loss(s,t,temperature)
        lp = F.log_softmax(s/temperature,dim=-1)
        pt = F.softmax(t/temperature,dim=-1)
        manual = (pt*(pt.log()-lp)).sum(-1).mean()*temperature**2
        torch.testing.assert_close(kd,manual)
        kd.backward()
        expected = temperature*(F.softmax(s.detach()/temperature,dim=-1)-pt.detach())/6
        torch.testing.assert_close(s.grad,expected)
        self.assertIsNone(t.grad)
        # Unflattened batchmean incorrectly multiplies the objective by sequence length.
        bad = F.kl_div(lp,pt.detach(),reduction='batchmean')*temperature**2
        torch.testing.assert_close(bad,3*kd.detach())

    def test_causality(self):
        torch.manual_seed(0)
        m = d.GPT(322,d.config(64)).eval()
        x = torch.randint(322,(2,12)); y = x.clone(); y[:,7:]=(y[:,7:]+1)%322
        torch.testing.assert_close(m(x)[:,:7],m(y)[:,:7],atol=1e-6,rtol=1e-6)

    def test_evaluate_preserves_state(self):
        m = d.GPT(322,d.config(64))
        data = torch.tensor(self.tok.encode(d.build_corpus(300,999)))
        for training in [False,True]:
            m.train(training)
            a=d.evaluate(m,data,2,'cpu',iters=2)
            b=d.evaluate(m,data,2,'cpu',iters=2)
            self.assertEqual(m.training,training)
            self.assertEqual(a,b)

    def test_same_student_initialization(self):
        a=d.make_model(64,7,'cpu'); b=d.make_model(64,7,'cpu')
        self.assertEqual(d.state_hash(a),d.state_hash(b))


if __name__ == '__main__':
    unittest.main(verbosity=2)
