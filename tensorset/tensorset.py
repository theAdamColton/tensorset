from typing import Union
import torch


def _validate_columns(tensorsets):
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


def _col_apply(
    tensorsets,
    func,
    **kwargs,
):
    """
    make a new tensorset by running func across each columns of all tensorsets
    """
    _validate_columns(tensorsets)
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


def cat(tensorsets, dim):
    """
    concatenate tensorsequences along dim

    returns a new TensorSet
    """
    return _col_apply(tensorsets, torch.cat, dim=dim)


def stack(tensorsets, dim=0):
    """
    stack n tensorsequences, adding a new dimension at stack_dim

    returns a new TensorSet
    """
    return _col_apply(tensorsets, torch.stack, dim=dim)


def stack_nt(tensorsets):
    """
    stack tensorsets using nested tensors

    returns a new TensorSet
    """
    return _col_apply(tensorsets, torch.nested.as_nested_tensor)


class _IlocIndexer:
    def __init__(self, tensorsequence):
        self.tensorsequence = tensorsequence

    def __getitem__(self, key):
        return TensorSet(
            *(c[key] for c in self.tensorsequence.columns),
            **{k: c[key] for k, c in self.tensorsequence.named_columns.items()},
        )


def _repr_column_shape(col):
    if col.is_nested:
        ndim = col.ndim

        s = "nested_tensor.Size(["
        for i in range(ndim):
            try:
                size = str(col.size(i))
            except:
                size = "irregular"
            s += size
            if i < ndim - 1:
                s += ", "
        s += "])"
        return s

    return str(col.shape)


class TensorSet:
    """
    A simple container of tensors in columns and named_columns
    """

    def __init__(
        self,
        *columns: torch.Tensor,
        **named_columns: torch.Tensor,
    ):
        self.columns = list(columns)
        self.named_columns = named_columns
        for c in self.all_columns:
            if not isinstance(c, torch.Tensor):
                raise ValueError(f"{type(c)} is not a torch.Tensor")

    def __repr__(self):
        s = "TensorSet(\n"
        if len(self.columns) > 0:
            s += "  columns:\n"

            s += "\n".join(
                f"    index: {i}, shape: {_repr_column_shape(c)}, dtype: {c.dtype}"
                for i, c in enumerate(self.columns)
            )
            s += "\n"
        if len(self.named_columns) > 0:
            s += "  named_columns:\n"
            s += "\n".join(
                f"    name: {n}, shape: {_repr_column_shape(c)}, dtype: {c.dtype}"
                for n, c in self.named_columns.items()
            )
            s += "\n"
        s += ")"
        return s

    def size(self, index: int):
        size = None
        for c in self.all_columns:
            if size is None:
                size = c.size(index)
            elif c.size(index) != size:
                raise ValueError(
                    f"Given dimension {index} is irregular and does not have a size."
                )
        return size

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
            raise ValueError(f"Key {key} is not an acceptable column accessor!")

    def __setitem__(self, key: Union[str, int], item: torch.Tensor):
        """
        Assign to a new or existing column of this tensorset
        """
        if isinstance(key, int):
            self.columns[key] = item
        elif isinstance(key, str):
            self.named_columns[key] = item

    def __delitem__(self, key: Union[str, int]):
        """
        deletes item based on column
        """
        if isinstance(key, int):
            self.columns = self.columns.drop(key)
        elif isinstance(key, str):
            self.columns = self.columns.drop(key)

    def __iter__(self):
        """
        returns an iterator
        """
        return iter(self.all_columns)

    def __len__(self):
        """
        return number of columns
        """
        return len(self.all_columns)

    def to_device(self, device, **kwargs):
        """
        in-place
        """
        self.columns = [c.to(device, **kwargs) for c in self.columns]
        self.named_columns = {
            k: c.to(device, **kwargs) for k, c in self.named_columns.items()
        }
        return self

    def pad(
        self,
        amount: int,
        dim: int,
        value=None,
        value_dict=None,
    ):
        """
        Pad all columns of this tensorsequence along dim

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
            pad_shape[dim] = amount

            pad_col = full(pad_shape, get_value(i))
            padding.append(pad_col)

        named_padding = {}
        for k, c in self.named_columns.items():
            pad_shape = list(c.shape)
            pad_shape[dim] = amount
            pad_col = full(pad_shape, get_value(k))
            named_padding[k] = pad_col
        padding = TensorSet(
            *padding,
            **named_padding,
        )
        return cat([self, padding], dim)
