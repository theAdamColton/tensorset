import unittest
import torch
from ..tensorsequence import TensorSequence


class TestTensorSet(unittest.TestCase):
    def test_validate_input_columns(self):
        def assert_value_error(do):
            occurred = False
            try:
                do()
            except ValueError:
                occurred = True
            self.assertTrue(occurred, "value error did not occurr as expected")

        c1 = torch.randn(3, 7, 9)
        c2 = torch.randn(3, 10, 3)

        # incompatible along dim 1
        assert_value_error(lambda: TensorSequence([c1, c2], sequence_dim=1))
        assert_value_error(lambda: TensorSequence([c1], {"c2": c2}, sequence_dim=1))

        # fine along dim 0
        TensorSequence([c1, c2], sequence_dim=0)
        TensorSequence([c1], {"c2": c2}, sequence_dim=0)

    def test_stack(self):
        c1 = torch.randn(7, 5, 3, 1, 1, 1)
        c2 = torch.randn(7, 5, 3, 4)
        c3 = torch.randn(7, 5, 3) > 0

        ts = TensorSequence([c1, c2], {"c3": c3}, 2)

        self.assertEqual(ts.sequence_length, 3)
        self.assertEqual(ts.leading_shape, (7, 5, 3))

        stacked = TensorSequence.stack([ts, ts, ts])

        self.assertEqual(3, stacked.sequence_dim)
        self.assertEqual((3, 7, 5, 3), stacked.leading_shape)
