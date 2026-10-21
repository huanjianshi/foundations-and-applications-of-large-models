"""Preflight tests: geometry, scoring, verifier boundaries, masks and stop gates."""
import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import hw6_rsi_localize as h


class TestLogic(unittest.TestCase):
    def test_grid_and_iou(self):
        self.assertEqual(h.cell_of([64, 64, 64, 64], 4), (1, 1))
        self.assertEqual(h.cell_of([255, 255, 256, 256], 4), (3, 3))
        self.assertEqual(h.cell_of([32, 32, 32, 32], 8), (1, 1))
        self.assertEqual(h.cell_box((1, 2), 4), [128, 64, 192, 128])
        self.assertEqual(h.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(h.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_parse_and_ambiguity(self):
        for s in ['第 2 行第 3 列', 'r2c3', 'R2 C3', '2,3', '2，3']:
            self.assertEqual(h.parse_cell(s, 4), (1, 2), s)
        for s in ['第0行第1列', 'r5c1', '第2行第3列，第3行第2列', '不知道']:
            self.assertIsNone(h.parse_cell(s, 4), s)
        self.assertEqual(h.parse_cell('r8c8', 8), (7, 7))
        self.assertIsNone(h.parse_cell('r8c8', 4))

    def test_qa_scoring(self):
        self.assertIsNone(h.parse_exist('图里有没有红色圆形？'))
        self.assertFalse(h.parse_exist('没有。'))
        self.assertFalse(h.parse_exist('No, there are no red circles.'))
        self.assertTrue(h.parse_exist('Yes, there is one.'))
        self.assertIsNone(h.parse_exist('Yes, there are no blue squares.'))
        self.assertEqual(h.parse_count('两个。'), 2)
        self.assertEqual(h.parse_count('There are two.'), 2)
        self.assertEqual(h.parse_count('1.0.'), 1)
        self.assertEqual(h.parse_count('2.00个。'), 2)
        self.assertIsNone(h.parse_count('1.5个'))
        self.assertIsNone(h.parse_count('-1个'))
        self.assertIsNone(h.parse_count('2个或者3个'))
        self.assertIsNone(h.parse_count('有几个？'))

    def test_verifier_without_gt_and_shape_limit(self):
        im = h.render([['红圆', [20, 20, 50, 50]], ['红方', [190, 190, 220, 220]]])
        self.assertEqual(len(h.color_components(im, '红')), 2)
        passed, _, evidence = h.verify(im, '红圆', 'loc', '第4行第4列', 4)
        self.assertTrue(passed)  # Known false acceptance: same color, wrong shape.
        self.assertFalse(evidence['shape_aware'])
        self.assertFalse(h.verify(im, '红圆', 'loc', '第2行第2列', 4)[0])
        self.assertFalse(h.verify(im, '红圆', 'loc', '无效答案', 4)[0])

    def test_fixed_data_and_disjoint_test(self):
        gt = h.load_gt()
        qs = h.questions(gt, 4)
        self.assertEqual(len(qs), 15)
        self.assertEqual([q['type'] for q in qs].count('loc'), 8)
        self.assertEqual(qs[0]['gold'], '第1行第2列')
        self.assertEqual(gt['e00'], [['红方', [65, 43, 106, 84]]])
        held = h.questions(gt, 4, 'heldout')
        self.assertEqual(len(held), 30)
        self.assertFalse({q['img'] for q in held} & {q['img'] for q in qs})
        self.assertEqual(sum(q['gold'] == '没有' for q in held), 3)
        qs8 = h.questions(gt, 8)
        self.assertEqual([(q['img'], q['target']) for q in qs], [(q['img'], q['target']) for q in qs8])

    def test_pause_thresholds(self):
        cache = h.ROOT/'.qa'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(h.ROOT.resolve()))
            args = Namespace(output=Path(folder), grid=4)
            def metrics(total, loc, valid=8):
                return [{'type': 'total', 'correct': total, 'n': 15, 'accuracy': total/15},
                        {'type': 'loc', 'correct': loc, 'n': 8, 'accuracy': loc/8, 'format_valid': valid}]
            h.gate(args, metrics(8, 4), 'seed')
            for total, loc in [(7, 4), (8, 3)]:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(h.QualityPause):
                    h.gate(args, metrics(total, loc), 'seed')
            h.gate(args, metrics(12, 6), 'final')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(h.QualityPause):
                h.gate(args, metrics(11, 6), 'final')

    def test_user_authorized_low_accuracy_continues(self):
        cache = h.ROOT/'.qa'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(h.ROOT.resolve()))
            args = Namespace(output=Path(folder), grid=4, accept_low_accuracy=True)
            metrics = [{'type': 'total', 'correct': 0, 'n': 15, 'accuracy': 0},
                       {'type': 'loc', 'correct': 0, 'n': 8, 'accuracy': 0, 'format_valid': 0}]
            h.gate(args, metrics, 'seed')
            self.assertEqual(h.read(Path(folder)/'quality_gate.json')['status'], 'ACCEPTED_LOW_ACCURACY')


if __name__ == '__main__':
    unittest.main(verbosity=2)
