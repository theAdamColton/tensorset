import unittest
import torch
from .. import tensorset as ts


class TestTensorSet(unittest.TestCase):
    def assert_error(self, do):
        occurred = False
        try:
            do()
        except:
            occurred = True
        self.assertTrue(occurred, "value error did not occurr as expected")

    def test_invalid_cat(self):
        b = 3
        x = ts.TensorSet(
            torch.empty(b, 7),
            torch.empty(b, 4),
        )

        y = ts.TensorSet(torch.empty(b, 10), torch.empty(b, 5))

        # cat is incompatible along dim 0
        self.assert_error(lambda: ts.cat([x, y], 0))

        # fine along dim 1
        ts.cat([x, y], 1)

    def test_invalid_cat_trailing(self):
        b = 3
        z = 16
        x = ts.TensorSet(
            torch.empty(b, 7, z),
            torch.empty(b, 4),
        )

        y = ts.TensorSet(torch.empty(b, 10, z + 1), torch.empty(b, 5))
        self.assert_error(lambda: ts.cat([x, y], 1))

    def test_stack(self):
        sequence_dim = 2
        c1 = torch.randn(7, 5, 3, 1, 1, 1)
        c2 = torch.randn(7, 5, 3, 4)
        c3 = torch.randn(7, 5, 3) > 0
        seq = ts.TensorSet(
            c1,
            c2,
            c3=c3,
        )

        self.assertEqual(seq.size(sequence_dim), 3)
        self.assertEqual(seq.size(0), 7)
        self.assertEqual(seq.size(1), 5)
        self.assertEqual(seq.size(2), 3)

        batch_size = 13
        stacked = ts.stack(list(seq for _ in range(batch_size)), 0)

        new_sequence_dim = sequence_dim + 1
        self.assertEqual(3, stacked.size(new_sequence_dim))
        self.assertEqual(batch_size, stacked.size(0))
        self.assertEqual(7, stacked.size(1))
        self.assertEqual(5, stacked.size(2))
        self.assertEqual(3, stacked.size(3))

        self.assertTrue(torch.equal(c1, stacked[0][0]))

    def test_iloc_dim0(self):
        c0 = torch.randn(10, 100)
        c1 = torch.randn(10, 100, 7)
        seq = ts.TensorSet(c0, c1)

        ts0 = seq.iloc[0]

        self.assertTrue(torch.equal(ts0.columns[0], c0[0]))

    def test_iloc_dim1(self):
        # batch, channel, sequence, z
        c0 = torch.randn(3, 2, 100)
        c1 = torch.randn(3, 2, 100, 4)
        x = ts.TensorSet(c0, c1)
        self.assertEqual(x.size(0), 3)
        self.assertEqual(x.size(1), 2)
        self.assertEqual(x.size(2), 100)

        ts0 = x.iloc[0]

        self.assertTrue(torch.equal(ts0[0], c0[0]))
        self.assertTrue(torch.equal(ts0[1], c1[0]))

        ts2 = x.iloc[2]

        self.assertTrue(torch.equal(ts2[0], c0[2]))
        self.assertTrue(torch.equal(ts2[1], c1[2]))

    def test_index_column(self):
        c0 = torch.randn(2, 55)
        c1 = torch.randn(2, 55, 3)
        x = ts.TensorSet(c0, c1)
        self.assertTrue(torch.equal(c0, x[0]))
        self.assertTrue(torch.equal(c1, x[1]))

    def test_index_column_by_name(self):
        c0 = torch.randn(2, 55)
        c1 = torch.randn(2, 55, 3)
        seq = ts.TensorSet(c0=c0, c1=c1)
        self.assertTrue(torch.equal(c0, seq["c0"]))
        self.assertTrue(torch.equal(c1, seq["c1"]))

    def test_cat(self):
        dim = 1
        c11 = torch.randn(2, 7)
        c21 = torch.randn(2, 7, 5)
        ts1 = ts.TensorSet(c11, c21)

        c12 = torch.randn(2, 32)
        c22 = torch.randn(2, 32, 5)
        ts2 = ts.TensorSet(c12, c22)

        tscat = ts.cat((ts1, ts2), dim)
        self.assertTrue(torch.equal(torch.cat((c11, c12), 1), tscat[0]))
        self.assertTrue(torch.equal(torch.cat((c21, c22), 1), tscat[1]))

    def test_pad_value(self):
        dim = 1
        c0 = torch.zeros(10, 1)
        c1 = torch.zeros(10, 1, 7)
        seq = ts.TensorSet(c0, c1)
        padded = seq.pad(15, dim, 1.0)
        self.assertEqual(padded.size(dim), 16)
        self.assertTrue(torch.all(padded[0][:, 1:] == 1.0))
        self.assertTrue(torch.all(padded[1][:, 1:] == 1.0))

    def test_pad_value_dict(self):
        dim = 1
        c0 = torch.zeros(8, 3)
        c1 = torch.zeros(8, 3, 1)
        seq = ts.TensorSet(c1=c0, c2=c1)
        padded = seq.pad(17, dim, value_dict=dict(c1=1.0, c2=2.0))
        self.assertEqual(padded.size(dim), 20)
        self.assertTrue(torch.all(padded["c1"][:, 3:] == 1.0))
        self.assertTrue(torch.all(padded["c2"][:, 3:] == 2.0))

    def test_to_device(self):
        c0 = torch.zeros(8, 3)
        c1 = torch.zeros(8, 3, 1)
        seq = ts.TensorSet(c1=c0, c2=c1)
        self.assertEqual(8, seq.size(0))
        self.assertEqual(3, seq.size(1))
        seq = seq.to_device("cpu")
        self.assertEqual(8, seq.size(0))
        self.assertEqual(3, seq.size(1))

    def test_stack_nt(self):
        seq1 = ts.TensorSet(torch.empty(10, 13), torch.empty(10, 13, 16))
        seq2 = ts.TensorSet(torch.empty(10, 4), torch.empty(10, 4, 16))
        stacked = ts.stack_nt((seq1, seq2))
        c1, c2 = stacked.columns
        self.assertEqual(2, c1.size(0))
        self.assertEqual(2, c2.size(0))
        self.assertEqual(16, c2.size(-1))

    def test_repr_nt(self):
        seq1 = ts.TensorSet(torch.empty(10, 13), torch.empty(10, 13, 16))
        seq2 = ts.TensorSet(torch.empty(10, 4), torch.empty(10, 4, 16))
        stacked = ts.stack_nt((seq1, seq2))
        s = repr(stacked)
        print(s)

    def test_size(self):
        seq = ts.TensorSet(
            torch.empty(10, 13, 32),
            torch.empty(10, 13, 16),
        )
        self.assertEqual(10, seq.size(0))
        self.assertEqual(13, seq.size(1))

        did_raise = False
        try:
            seq.size(2)
        except:
            did_raise = True
        self.assertTrue(did_raise)

    def test_cat_tensorsets(self):
        seq1 = ts.TensorSet(
            torch.empty(10, 13, 32),
            torch.empty(10, 13, 16),
        )
        seq2 = ts.TensorSet(torch.empty(10, 13, 100), torch.empty(10, 13, 20))
        catted = ts.cat((seq1, seq2), -1)
        self.assertEqual(10, catted.size(0))
        self.assertEqual(13, catted.size(1))
        self.assertEqual(132, catted[0].size(-1))
        self.assertEqual(36, catted[1].size(-1))

    def test_set_item(self):
        seq = ts.TensorSet()
        seq["asdf"] = torch.empty(1, 1, 1)
        self.assertEqual(1, seq.num_columns)
        seq2 = ts.TensorSet(torch.empty(10, 10))

        seq2[0] = torch.zeros(1, 1, 1)
        self.assertEqual(seq2[0].shape, torch.zeros(1, 1, 1).shape)

    def test_iter(self):

        seq = ts.TensorSet(
            torch.empty(1, 1, 1),
            torch.empty(
                1,
                1,
            ),
            torch.empty(
                1,
            ),
        )
        iterator = iter(seq)
        self.assertEqual(torch.empty(1, 1, 1).shape, next(iterator).shape)
        self.assertEqual(
            torch.empty(
                1,
                1,
            ).shape,
            next(iterator).shape,
        )
        self.assertEqual(
            torch.empty(
                1,
            ).shape,
            next(iterator).shape,
        )

    def test_len(self):
        seq = ts.TensorSet(
            torch.empty(1, 1, 1),
            torch.empty(
                1,
                1,
            ),
            torch.empty(
                1,
            ),
        )
        self.assertEqual(len(seq), 3)
