from collections.abc import Iterable
from typing import Dict
import torch


def validate_tensor_sets_broadcastable(tensorsequences):
    if len(tensorsequences) == 0:
        return

    names = tensorsequences[0].named_columns.keys()

    for ts in tensorsequences:
        if ts.named_columns.keys() != names:
            raise ValueError("Different named columns between tensorsequences")

    num_columns = len(tensorsequences[0].columns)
    num_named_columns = len(tensorsequences[0].named_columns)
    for ts in tensorsequences:
        if len(ts.columns) != num_columns:
            raise ValueError("tensorsequencesdon't have matching number of columns")
        if len(ts.named_columns) != num_named_columns:
            raise ValueError(
                "tensorsequencesdon't have matching number of named_columns"
            )

    sequence_dim = tensorsequences[0].sequence_dim

    if not all(ts.sequence_dim == sequence_dim for ts in tensorsequences):
        raise ValueError("Tensor sets have different sequence dimensions!")

    if len(tensorsequences[0].all_columns) == 0:
        return

    # must have matching leading dimensions up to (but not including) the sequence dim
    shape = tensorsequences[0].all_columns[0].shape[:sequence_dim]

    for ts in tensorsequences:
        for c in ts.all_columns:
            if c.shape[:sequence_dim] != shape:
                raise ValueError("Tensor sets have mismatched leading dimensions!")


def validate_input_columns(columns, sequence_dim):
    if len(columns) == 0:
        return
    shape = columns[0].shape[: sequence_dim + 1]
    for c_i, c in enumerate(columns):
        c_shape = c.shape[: sequence_dim + 1]
        for i, (s, c_s) in enumerate(zip(shape, c_shape)):
            if s != c_s:
                raise ValueError(
                    f"column number {c_i} has incompatible shape at dimension {i}, {s} != {c_s}"
                )


class TensorSequence:
    """
    A small wrapper allowing manipulation of a set of columns,
    each column with the same sequence length (rows)

    """

    def __init__(
        self,
        columns: Iterable[torch.Tensor] = [],
        named_columns: Dict[str, torch.Tensor] = {},
        sequence_dim: int = 0,
    ):
        """
        Columns do not have to have the same dype or device

        Each column has matching leading dimensions. This means that up to the sequence dimension, all of the shapes of all of the columns have to be matching.

        For example, if sequence dim = 1, all columns must be shape (B, S, ...)
         where the ellipses indicates any number of trailing dims which don't have to match
         and the B dimension is matching across all columns.
        """
        validate_input_columns(
            list(columns) + list(named_columns.values()), sequence_dim
        )
        self.columns = list(columns)
        self.named_columns = named_columns
        self.sequence_dim = sequence_dim

    @property
    def all_columns(self):
        return self.columns + list(self.named_columns.values())

    @property
    def sequence_length(self):
        if len(self.all_columns) == 0:
            return 0
        return self.all_columns[0].shape[self.sequence_dim]

    @property
    def leading_shape(self):
        if len(self.all_columns) == 0:
            return tuple()
        return self.all_columns[0].shape[: self.sequence_dim + 1]

    @property
    def num_columns(self):
        return len(self.all_columns)

    def __getitem__(self, key):
        return TensorSequence(
            list(c[key] for c in self.columns),
            {k: c[key] for k, c in self.named_columns.items()},
        )

    @staticmethod
    def _run_func(tensorsequences, func, axis=None, new_sequence_dim=None):
        """
        make a new tensorsequence by running func across each columns of all tensorsequences
        """
        validate_tensor_sets_broadcastable(tensorsequences)
        sequence_dim = tensorsequences[0].sequence_dim

        if axis is None:
            axis = sequence_dim

        num_columns = len(tensorsequences[0].columns)
        columns = []
        for i in range(num_columns):
            columns.append(func([ts.columns[i] for ts in tensorsequences], axis))

        names = tensorsequences[0].named_columns.keys()
        named_columns = {}
        for name in names:
            named_columns[name] = func(
                [ts.named_columns[name] for ts in tensorsequences], axis
            )

        if new_sequence_dim is None:
            new_sequence_dim = sequence_dim

        return TensorSequence(columns, named_columns, sequence_dim=new_sequence_dim)

    @staticmethod
    def cat(tensorsequences):
        """
        Cat tensorsequencesalong their sequence dimensions
        """
        return TensorSequence._run_func(
            tensorsequences, torch.cat, axis=None, new_sequence_dim=None
        )

    @staticmethod
    def stack(tensorsequences):
        """
        stack tensorsequencesalong dim 0, adding a new leading dimension
        """
        sequence_dim = tensorsequences[0].sequence_dim
        return TensorSequence._run_func(
            tensorsequences, torch.stack, 0, sequence_dim + 1
        )

    def pad(self, amount: int, value=-1):
        """
        Pad all columns of this tensorsequence along the sequence dimension
        """
        assert amount >= 0
        padding = []
        full = lambda shape: torch.full(
            shape,
            value,
            dtype=c.dtype,
            device=c.device,
        )
        for c in self.columns:
            pad_shape = list(c.shape)
            pad_shape[self.sequence_dim] = amount

            pad_col = full(pad_shape)
            padding.append(pad_col)

        named_padding = {}
        for k, c in self.named_columns.items():
            pad_shape = list(c.shape)
            pad_shape[self.sequence_dim] = amount
            pad_col = full(pad_shape)
            named_padding[k] = pad_col
        padding = TensorSequence(padding, named_padding, self.sequence_dim)
        return TensorSequence.cat([self, padding])

    def to_device(self, device):
        """
        in-place
        """
        self.columns = [c.to(device) for c in self.columns]
        self.named_columns = {k: c.to(device) for k, c in self.named_columns.items()}
        return self
