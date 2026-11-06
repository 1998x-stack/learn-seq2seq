"""Regression tests for the m03 ignore_index teaching example.

Run: python -m unittest discover -s tests -p 'test_m03_padding.py'
"""
import unittest

import torch
import torch.nn.functional as F


class PaddingLossTests(unittest.TestCase):
    def test_ignored_positions_do_not_change_retained_loss(self):
        logits = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]])
        targets = torch.tensor([0, 1, 2])
        masked = targets.clone()
        masked[1] = -100
        expected = F.cross_entropy(logits[[0, 2]], targets[[0, 2]])
        actual = F.cross_entropy(logits, masked, ignore_index=-100)
        self.assertTrue(torch.allclose(actual, expected))

    def test_ignoring_hard_example_can_reduce_loss(self):
        logits = torch.tensor([[8., 0.], [8., 0.]])
        targets = torch.tensor([0, 1])
        masked = torch.tensor([0, -100])
        self.assertLess(F.cross_entropy(logits, masked, ignore_index=-100),
                        F.cross_entropy(logits, targets))

    def test_ignoring_easy_example_can_increase_loss(self):
        logits = torch.tensor([[8., 0.], [8., 0.]])
        targets = torch.tensor([0, 1])
        masked = torch.tensor([-100, 1])
        self.assertGreater(F.cross_entropy(logits, masked, ignore_index=-100),
                           F.cross_entropy(logits, targets))


if __name__ == "__main__":
    unittest.main()
