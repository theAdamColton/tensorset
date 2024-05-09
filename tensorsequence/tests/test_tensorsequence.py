import unittest
import torch
from .. import tensorsequence as ts


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
        assert_value_error(lambda: ts.TensorSequence(c1, c2, sequence_dim=1))
        assert_value_error(lambda: ts.TensorSequence(c1, c2=c2, sequence_dim=1))

        # fine along dim 0
        ts.TensorSequence(c1, c2, sequence_dim=0)
        ts.TensorSequence(c1, c2=c2, sequence_dim=0)

    def test_stack(self):
        c1 = torch.randn(7, 5, 3, 1, 1, 1)
        c2 = torch.randn(7, 5, 3, 4)
        c3 = torch.randn(7, 5, 3) > 0

        seq = ts.TensorSequence(c1, c2, c3=c3, sequence_dim=2)

        self.assertEqual(seq.sequence_length, 3)
        self.assertEqual(seq.leading_shape, (7, 5, 3))

        batch_size = 13
        stacked = ts.stack((seq for _ in range(batch_size)))

        self.assertEqual(3, stacked.sequence_dim)
        self.assertEqual((batch_size, 7, 5, 3), stacked.leading_shape)
        self.assertTrue(torch.equal(c1, stacked[0][0]))

    def test_iloc_dim0(self):
        c0 = torch.randn(10, 100)
        c1 = torch.randn(10, 100, 7)
        seq = ts.TensorSequence(c0, c1, sequence_dim=1)

        ts0 = seq.iloc[0]

        self.assertIsInstance(ts0, ts.TensorSet)
        self.assertTrue(torch.equal(ts0.columns[0], c0[0]))

    def test_iloc_dim1(self):
        # batch, channel, sequence, z
        c0 = torch.randn(3, 2, 100)
        c1 = torch.randn(3, 2, 100, 4)
        seq = ts.TensorSequence(c0, c1, sequence_dim=2)
        self.assertEqual(seq.sequence_length, 100)

        ts0 = seq.iloc[0]

        self.assertIsInstance(ts0, ts.TensorSet)
        self.assertTrue(torch.equal(ts0.columns[0], c0[0]))
        self.assertTrue(torch.equal(ts0.columns[1], c1[0]))

        ts2 = seq.iloc[2]

        self.assertIsInstance(ts2, ts.TensorSet)
        self.assertTrue(torch.equal(ts2.columns[0], c0[2]))
        self.assertTrue(torch.equal(ts2.columns[1], c1[2]))

    def test_index_column(self):
        c0 = torch.randn(2, 55)
        c1 = torch.randn(2, 55, 3)
        seq = ts.TensorSequence(c0, c1, sequence_dim=1)
        self.assertTrue(torch.equal(c0, seq[0]))
        self.assertTrue(torch.equal(c1, seq[1]))

    def test_index_column_by_name(self):
        c0 = torch.randn(2, 55)
        c1 = torch.randn(2, 55, 3)
        seq = ts.TensorSequence(c0=c0, c1=c1, sequence_dim=1)
        self.assertTrue(torch.equal(c0, seq["c0"]))
        self.assertTrue(torch.equal(c1, seq["c1"]))

    def test_cat(self):
        c11 = torch.randn(2, 7)
        c21 = torch.randn(2, 7, 5)
        ts1 = ts.TensorSequence(c11, c21, sequence_dim=1)

        c12 = torch.randn(2, 32)
        c22 = torch.randn(2, 32, 5)
        ts2 = ts.TensorSequence(c12, c22, sequence_dim=1)

        tscat = ts.cat((ts1, ts2))
        self.assertTrue(torch.equal(torch.cat((c11, c12), 1), tscat[0]))
        self.assertTrue(torch.equal(torch.cat((c21, c22), 1), tscat[1]))

    def test_pad_value(self):
        c0 = torch.zeros(10, 1)
        c1 = torch.zeros(10, 1, 7)
        seq = ts.TensorSequence(c0, c1, sequence_dim=1)
        padded = seq.pad(15, 1.0)
        self.assertEqual(padded.sequence_length, 16)
        self.assertTrue(torch.all(padded[0][:, 1:] == 1.0))
        self.assertTrue(torch.all(padded[1][:, 1:] == 1.0))

    def test_pad_value_dict(self):
        c0 = torch.zeros(8, 3)
        c1 = torch.zeros(8, 3, 1)
        seq = ts.TensorSequence(c1=c0, c2=c1, sequence_dim=1)
        padded = seq.pad(17, value_dict=dict(c1=1.0, c2=2.0))
        self.assertEqual(padded.sequence_length, 20)
        self.assertTrue(torch.all(padded["c1"][:, 3:] == 1.0))
        self.assertTrue(torch.all(padded["c2"][:, 3:] == 2.0))

    def test_to_device(self):
        c0 = torch.zeros(8, 3)
        c1 = torch.zeros(8, 3, 1)
        seq = ts.TensorSequence(c1=c0, c2=c1, sequence_dim=1)
        self.assertEqual((8, 3), seq.leading_shape)
        self.assertEqual(3, seq.sequence_length)
        self.assertEqual(1, seq.sequence_dim)
        seq = seq.to_device("cpu")
        self.assertEqual((8, 3), seq.leading_shape)
        self.assertEqual(3, seq.sequence_length)
        self.assertEqual(1, seq.sequence_dim)

    def test_stack_nt(self):
        seq1 = ts.TensorSequence(
            torch.empty(10, 13), torch.empty(10, 13, 16), sequence_dim=1
        )
        seq2 = ts.TensorSequence(
            torch.empty(10, 4), torch.empty(10, 4, 16), sequence_dim=1
        )
        stacked = ts.stack_nt((seq1, seq2))
        c1, c2 = stacked.columns
        self.assertEqual(2, c1.size(0))
        self.assertEqual(2, c2.size(0))
        self.assertEqual(16, c2.size(-1))
