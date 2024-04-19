# tensorsequence

tensorsequecne lets you perform operations on related sequences using a unified `TensorSequence` object.

It aims to reduce the complexity of pytorch code when using sequences commonly used as inputs to transformer models:
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
is_token_whitespace_mask = (input_ids == 0) | (input_ids == 1)# Shape: batch_size, sequence_length 

# These tensors would be used like this:
# logits = transformer_model(input_embeds, key_pad_mask, is_token_whitespace_mask)
```

Notice that any place where these tensors need to be truncated or stacked or concatenated has repeated tediously code:

```python
def truncate_inputs(hidden_states, key_pad_mask, is_token_whitespace_mask, length):
  hidden_states = hidden_states[:, :length]
  key_pad_mask= key_pad_mask[:, :length]
  is_token_whitespace_mask= is_token_whitespace_mask[:, :length, :length]
  return hidden_states, key_pad_mask, attention_mask

truncated_inputs = truncate_inputs(hidden_states, key_pad_mask, is_token_whitespace_mask, length)
```


Notice that `input_ids`, `input_embeds`, `key_pad_mask`, and `is_whitespace_mask` are all related.
They all have matching leading dimensions for batch_size and sequence length. 
TensorSequence is a container for these related multi-dimensional sequences.

```python
from tensorsequence import TensorSequence
inputs = TensorSequence(input_ids, input_embeds, key_pad_mask, is_whitespace_mask, sequence_dim=1)
```

Common operations are now much more ergonomic.

```
truncated_inputs = inputs[:, :length]
```

You can stack related TensorSets to create larger batches

```python
sequence_length = 20
sequence_1 = TensorSet(torch.randn(sequence_length, 512), torch.randn(sequence_length, 1024), sequence_dim=0)
sequence_2 = TensorSet(torch.randn(sequence_length, 512), torch.randn(sequence_length, 1024), sequence_dim=0)
batch = TensorSet.stack(sequence_1, sequence_2)

print(batch.sequence_length) # Prints 20
print(batch.leading_shape[0]) # Prints 2
```
