"""Boundary tests for paired stopping and source optimizer/RNG preservation."""
import unittest

from .common import PairPatience, TRIALS


class StoppingTests(unittest.TestCase):
    def make(self):
        return PairPatience(dict(zip(TRIALS,(.614,.613))))

    def test_both_plateau_still_respect_minimum(self):
        rule = self.make()
        for _ in range(9):
            self.assertFalse(rule.update(dict(zip(TRIALS,(.613,.612)))))
        self.assertTrue(rule.update(dict(zip(TRIALS,(.613,.612)))))

    def test_one_arm_improving_keeps_both_running(self):
        rule = self.make()
        for epoch in range(15):
            self.assertFalse(rule.update(dict(zip(TRIALS,(.61,.613+.001*(epoch+1))))))
        for _ in range(7):
            self.assertFalse(rule.update(dict(zip(TRIALS,(.61,.627)))))
        self.assertTrue(rule.update(dict(zip(TRIALS,(.61,.627)))))

    def test_resume_preserves_patience(self):
        rule = self.make()
        for _ in range(7):
            rule.update(dict(zip(TRIALS,(.61,.61))))
        restored = PairPatience.restore(rule.state())
        for _ in range(3):
            self.assertEqual(rule.update(dict(zip(TRIALS,(.61,.61)))),
                             restored.update(dict(zip(TRIALS,(.61,.61)))))
        self.assertEqual(rule.state(),restored.state())

    def test_tiny_improvement_does_not_reset_plateau(self):
        rule = self.make()
        rule.update(dict(zip(TRIALS,(.61405,.61305))))
        self.assertEqual(list(rule.bad.values()),[1,1])
        rule.update(dict(zip(TRIALS,(.6142,.6132))))
        self.assertEqual(list(rule.bad.values()),[0,0])

    def test_missing_nan_and_out_of_range_pair_rejected(self):
        for values in ({TRIALS[0]:.6},dict(zip(TRIALS,(float('nan'),.6))),dict(zip(TRIALS,(1.1,.6)))):
            with self.assertRaises(ValueError):
                self.make().update(values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
