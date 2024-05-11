# tensorset

tensorset is a pytorch library that lets you perform operations on related sequences using a unified `TensorSet` object.

It aims to reduce the complexity of using multiple related sequences. Sequences like these are very commonly used as inputs to a transformer model:
```python
import torch
from torch import nn

batch_size = 8
sequence_length = 1024
vocab_size = 256
hidden_size = 768
pad_id = 255

token_embeddings = nn.Embedding(vocab_size, hidden_size)

input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
input_embeds = token_embeddings(input_ids) # Shape: batch_size, sequence_length, hidden_size
key_pad_mask = input_ids == pad_id # Shape: batch_size, sequence_length 
is_whitespace_mask = (input_ids == 0) | (input_ids == 1)# Shape: batch_size, sequence_length 

# These tensors would be used like this:
# logits = transformer_model(input_embeds, key_pad_mask, is_whitespace_mask)
```

Notice wherever these tensors are truncated or stacked or concatenated there will be tedious repetitive code like this:

```python
def truncate_inputs(input_ids, key_pad_mask, is_whitespace_mask, length):
  input_ids= input_ids[:, :length]
  key_pad_mask= key_pad_mask[:, :length]
  is_whitespace_mask= is_whitespace_mask[:, :length]
  return input_ids, key_pad_mask, is_whitespace_mask

truncated_inputs = truncate_inputs(input_ids, key_pad_mask, is_whitespace_mask, length=10)
```

This repetitive code can be avoided. `input_ids`, `input_embeds`, `key_pad_mask`, and `is_whitespace_mask` are all related.
They all have matching leading dimensions for batch_size and sequence length. 

TensorSet is a container for these related multi-dimensional sequences, making this kind of manipulation very easy and ergonomic.

```python
import tensorset as ts
length = 10
inputs = ts.TensorSet(
                input_ids=input_ids,
                input_embeds=input_embeds,
                key_pad_mask=key_pad_mask,
                is_whitespace_mask=is_whitespace_mask,
         )
truncated_inputs = inputs.iloc[:, :length]
print(truncated_inputs)
```

prints:
```
TensorSet(
  named_columns:
    name: input_ids, shape: torch.Size([8, 10]), dtype: torch.int64
    name: input_embeds, shape: torch.Size([8, 10, 768]), dtype: torch.float32
    name: key_pad_mask, shape: torch.Size([8, 10]), dtype: torch.bool
    name: is_whitespace_mask, shape: torch.Size([8, 10]), dtype: torch.bool
)
```


# Features

Stack related TensorSets to create larger batches

```python
sequence_length = 20
sequence_1 = ts.TensorSet(
                torch.randn(sequence_length, 512),
                torch.randn(sequence_length, 1024),
            )
sequence_2 = ts.TensorSet(
                torch.randn(sequence_length, 512),
                torch.randn(sequence_length, 1024),
            )
batch = ts.stack((sequence_1, sequence_2), 0)

print(batch.size(1)) # This is the sequence length, prints 20
print(batch.size(0)) # This is the batch size, prints 2
```

Pad TensorSets with a specific amount of padding along the sequence dimension
```python
sequence_length = 20
sequence = ts.TensorSet(
                torch.randn(sequence_length, 512),
                torch.randn(sequence_length, 1024),
            )
pad_value = -200
padded_sequence = sequence.pad(44, 0, pad_value) # add 44 dims of padding along dimension 0, of pad_value
print(padded_sequence.size(0)) # This is the new sequence length, prints 64
```

Stack TensorSets with irregular shape, using torch.nested
```python
# C, H, W pixel_values, and an additional binary mask
image1 = ts.TensorSet(
          pixel_values = torch.randn(3, 20, 305),
          mask = torch.randn(3, 20, 305) > 0,
        )
image2 = ts.TensorSet(
          pixel_values = torch.randn(3, 450, 200),
          mask = torch.randn(3, 450, 200) > 0,
        )
images = ts.stack_nt([image1, image2])
print(images)
```

output:
```
TensorSet(
  named_columns:
    name: pixel_values, shape: nested_tensor.Size([2, 3, irregular, irregular]), dtype: torch.float32
    name: mask, shape: nested_tensor.Size([2, 3, irregular, irregular]), dtype: torch.bool
)
```

# TODO

* Access by lists of columns
* Enable operations over irregular dims that are not supported yet by torch.nested, such as mean and index select
