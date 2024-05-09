from collections.abc import Iterable
from typing import Dict, Union
import torch


def _validate_columns(tensorsets):
    if len(tensorsets) == 0:
        return

    names = tensorsets[0].named_columns.keys()

    for ts in tensorsets:
        if ts.named_columns.keys() != names:
            raise ValueError("Different named columns between tensorsequences")

    num_columns = len(tensorsets[0].columns)
    num_named_columns = len(tensorsets[0].named_columns)
    for ts in tensorsets:
        if len(ts.columns) != num_columns:
            raise ValueError("tensorsequencesdon't have matching number of columns")
        if len(ts.named_columns) != num_named_columns:
            raise ValueError(
                "tensorsequencesdon't have matching number of named_columns"
            )


def _validate_tensor_sets_broadcastable(tensorsequences):
    if len(tensorsequences) == 0:
        return

    _validate_columns(tensorsequences)

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


def _validate_input_columns(columns, sequence_dim):
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


def _col_apply(
    tensorsets,
    func,
    **kwargs,
):
    """
    make a new tensorset by running func across each columns of all tensorsets
    """
    num_columns = len(tensorsets[0].columns)
    columns = []
    for i in range(num_columns):
        columns.append(func([ts.columns[i] for ts in tensorsets], **kwargs))

    names = tensorsets[0].named_columns.keys()
    named_columns = {}
    for name in names:
        named_columns[name] = func(
            [ts.named_columns[name] for ts in tensorsets], **kwargs
        )

    return TensorSet(*columns, **named_columns)


def cat(tensorsequences):
    """
    concatenate tensorsequences along the sequence dim

    returns a new TensorSequence with a longer sequence dimension
    """
    _validate_tensor_sets_broadcastable(tensorsequences)
    sequence_dim = tensorsequences[0].sequence_dim
    tensorsequences = _col_apply(tensorsequences, torch.cat, dim=sequence_dim)
    return TensorSequence(
        *tensorsequences.columns,
        sequence_dim=sequence_dim,
        **tensorsequences.named_columns,
    )


def stack(tensorsequences, stack_dim=0):
    """
    stack n tensorsequences along the sequence dim, adding a new dimension at stack_dim

    tensorsequences:
    each tensorsequence has shape: (... S ...) where S is sequence length

    returns a new TensorSequence with an additional leading dimension of size n
    the location of the new leading dimension is specified by stack_dim
    shape: (... N ... S ...)

    the default behavior is to place N at 0
    shape: (N ... S ...)
    """
    _validate_tensor_sets_broadcastable(tensorsequences)
    sequence_dim = tensorsequences[0].sequence_dim
    if stack_dim > sequence_dim:
        raise ValueError(
            f"dimension to use for stacking {stack_dim} must be less than sequence dim {sequence_dim}."
        )
    tensorsequences = _col_apply(tensorsequences, torch.stack, dim=stack_dim)
    return TensorSequence(
        *tensorsequences.columns,
        sequence_dim=sequence_dim + 1,
        **tensorsequences.named_columns,
    )


def stack_nt(tensorsets):
    """
    stack tensorsets along the sequence dim, using nested tensors.
    This avoids needing to use tensorsequences that have matching sequence lengths
    """
    _validate_columns(tensorsets)
    return _col_apply(tensorsets, torch.nested.as_nested_tensor)


class _IlocIndexer:
    def __init__(self, tensorsequence):
        self.tensorsequence = tensorsequence

    def __getitem__(self, key):
        return TensorSet(
            *(c[key] for c in self.tensorsequence.columns),
            **{k: c[key] for k, c in self.tensorsequence.named_columns.items()},
        )


class TensorSet:
    """
    A simple container of tensors in columns and named_columns
    """

    def __init__(
        self,
        *columns,
        **named_columns,
    ):
        self.columns = list(columns)
        self.named_columns = named_columns

    def __repr__(self):
        s = "TensorSet(\n"
        if len(self.columns) > 0:
            s += "  columns:\n"

            s += "\n".join(
                f"    index: {i}, shape: {c.shape}, dtype: {c.dtype}"
                for i, c in enumerate(self.columns)
            )
            s += "\n"
        if len(self.named_columns) > 0:
            s += "  named_columns:\n"
            s += "\n".join(
                f"    name: {n}, shape: {c.shape}, dtype: {c.dtype}"
                for n, c in self.named_columns.items()
            )
            s += "\n"
        s += ")"
        return s

    @property
    def all_columns(self):
        return self.columns + list(self.named_columns.values())

    @property
    def num_columns(self):
        return len(self.all_columns)

    @property
    def iloc(self):
        """
        Index into the rows of all of the columns of this tensorset.
        """
        return _IlocIndexer(self)

    def __getitem__(self, key: Union[str, int]):
        """
        Access the columns of this tensorset, using either integer keys (to access self.columns)
            or string keys (to access self.named_columns)
        """
        if isinstance(key, str):
            return self.named_columns[key]
        elif isinstance(key, int):
            return self.columns[key]
        else:
            raise ValueError(key)

    def __setitem__(self, key: Union[str, int], item: torch.Tensor):
        """
        Assign to a new or existing column of this tensorset
        """
        if isinstance(key, int):
            self.columns[key] = item
        elif isinstance(key, str):
            self.named_columns[key] = item

    def to_device(self, device):
        """
        in-place
        """
        self.columns = [c.to(device) for c in self.columns]
        self.named_columns = {k: c.to(device) for k, c in self.named_columns.items()}
        return self


class TensorSequence(TensorSet):
    """
    A small wrapper allowing manipulation of a set of columns,
    each column with the same leading shape. The leading shape
    of a tensor is it's dimensions up to and including the sequence length dimension.

    Because of the consistent leading dimensions, TensorSequence
    supports sequence oriented transforms like concatenation
    and stacking and padding.
    """

    def __init__(
        self,
        *columns,
        sequence_dim: int = 0,
        **named_columns,
    ):
        """
        Columns do not have to have the same dype or device

        Each column has matching leading dimensions. This means that up to the sequence dimension, all of the shapes of all of the columns have to be matching.

        For example, if sequence dim = 1, all columns must be shape (B, S, ...)
         where the ellipses indicates any number of trailing dims which don't have to match
         and the B dimension is matching across all columns.
        """
        columns = list(columns)
        _validate_input_columns(columns + list(named_columns.values()), sequence_dim)
        self.columns = columns
        self.named_columns = named_columns
        self.sequence_dim = sequence_dim

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

    def pad(self, amount: int, value=None, value_dict=None):
        """
        Pad all columns of this tensorsequence along the sequence dimension

        The value which is used for padding is specified by `value` or `value_dict`

        value: a single number
        value_dict: a dict which maps column_name to value.
            To associate a padding value with an unnamed column, use a integer key.
            Otherwise, use a string key.
        """
        assert amount >= 0
        padding = []
        full = lambda shape, value: torch.full(
            shape,
            value,
            dtype=c.dtype,
            device=c.device,
        )

        if value is None and value_dict is None:
            raise ValueError("either value or value_dict must be specified")

        get_value = lambda key: value if value is not None else value_dict[key]

        for i, c in enumerate(self.columns):
            pad_shape = list(c.shape)
            pad_shape[self.sequence_dim] = amount

            pad_col = full(pad_shape, get_value(i))
            padding.append(pad_col)

        named_padding = {}
        for k, c in self.named_columns.items():
            pad_shape = list(c.shape)
            pad_shape[self.sequence_dim] = amount
            pad_col = full(pad_shape, get_value(k))
            named_padding[k] = pad_col
        padding = TensorSequence(
            *padding,
            sequence_dim=self.sequence_dim,
            **named_padding,
        )
        return cat([self, padding])

    def __setitem__(self, key: Union[str, int], value: torch.Tensor):
        _validate_input_columns(self.all_columns + [value], self.sequence_dim)
        super().__setitem__(key, value)
