import unittest
import torch
from benchmarks.compare_model import difference


class DifferenceTests(unittest.TestCase):
    def test_equal(self):
        x = torch.tensor([1e-5, 2e-5])
        self.assertTrue(difference(x, x, .05)["passed"])

    def test_small_but_incorrect_gradient_fails(self):
        reference = torch.tensor([1e-5, 2e-5])
        self.assertFalse(difference(torch.zeros_like(reference), reference, .05)["passed"])

    def test_nan_fails(self):
        self.assertFalse(difference(torch.tensor([float("nan")]), torch.ones(1), .05)["passed"])


if __name__ == "__main__":
    unittest.main()
