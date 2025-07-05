# nanoGPT-recommender

Adaptation of [nanoGPT model](https://github.com/karpathy/nanoGPT) to a task of Amazon product recommendation for [Data Science Club PJATK kaggle challenge](https://www.kaggle.com/competitions/product-recommendation-challenge).

The idea is to treat purchases as sequences of tokens (item_ids), and based on user's current history we predict next item that will be bought. During training we take available sequences and shift them one left to make targets (next word prediction target).
Additionally this repo leverages [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip) embeddings to incorporate image data into the model.

Key changes to the original architecture:
- pack input sequences for more efficient training (70% of training dataset, ~600k rows, are sequences of length 2, so after making it a training example, it's just one input, one output token per row)
    - adapt causal attention mask, so independent samples don't influence each other, see [here](https://huggingface.co/blog/sirluk/llm-sequence-packing) for visual example of the attention masking that was needed
    - generate pos ids which start from 0 for each sample in a packed sequence, so positional info is reflected correctly during inference where samples will always have positional info starting from 0
- add [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip) embedding layer
    - lookup image embeddings for each item_id, then add them to the initial embedddings, so `tok_emb + pos_emb + siglip_proj` (where `siglip_proj = linear(siglip_emb)` projects embeddings to match the model's dimension)
    - kept SigLIP's embeddings frozen, left linear projection trainable

Additionaly there's a simplified version of training script `simple_train.py` and a notebook to prepare SigLIP embeddings.

# Submission journey

| Model Name | Score |
|------------|-------|
| 1. svd | 0.00245 |
| 2. siglip_knn | 0.00723 |
| 3. lightgcn | 0.04754 |
| 4. transformer | 0.05068 |
| 5. transformer_generative | 0.05955 |
| 6. siglip_oneshot | 0.06788 |
| 7. with_decay | 0.07198 |
| 8. all_data | 0.07401 |

1. `svd` matrix factorization baseline, used Claude's vectorized script for inference to end in a reasonable timeframe (~3h), not sure if it was correct though
2. `siglip_knn` embed product image with siglip, then take user's last bought product and find the closest one to it. Use [Faiss](https://github.com/facebookresearch/faiss) for fast embedding lookup
3. `lightgcn` some baseline graph model I found in PyTorch Geometric's [examples](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/lightgcn.py). Not sure if I (Claude) adapted everything correctly to this dataset.
4. `transformer` baseline transformer with sequence packing.
5. `transformer_generative` to get top 10 predictions as a baseline I was taking 10 items with highest logits on the next token inference. Other approach would be to generate 10 next tokens and use that as predictions. Here it improved the results significantly, later with siglip I tried top1/top3 sampling when generating tokens and it performed worse than doing it in oneshot from logits.
6. `siglip_oneshot` including SigLIP embeddings.
7. `with_decay` for previous runs I've been using a very minimal version of training script, here I've made a refactor to make it run faster and additionally added LR decay back from the original training script.
8. `all_data` putting back the validation dataset (10% of all entries) back into the training dataset, training for the same amount of steps as previous models.

# Trying out Flex Attention

In nanoGPT-flex I've used [Flex Attention](https://pytorch.org/blog/flexattention/) which provides a convenient interface for working with custom attention masking and also optimizes computation for spare masks by computing attention in blocks, which also avoids materializing the whole mask tensor.

[Here's](https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences) an example on how to combine document and prefixLM masking (prefix + causal + document masking). In our use case we're combining only document and causal masking and it looks like this

```
def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = sequence_ids[b, q_idx] == sequence_ids[b, kv_idx]
    return causal_mask & document_mask

block_mask = create_block_mask(document_causal_mask, b, None, t, t, device=device, BLOCK_SIZE=128)

(...)

y = flex_attention(q, k, v, block_mask=block_mask)
```

When creating block mask it seems that 128 is the lowest value that works, it might work with 64, but didn't for me ([see here](https://github.com/pytorch/pytorch/issues/133562)). With `batch_size=256`, there seems to be a small speedup compared to `scaled_dot_product_attention`.

# Trying out Nested Jagged Tensors

[NJTs](https://docs.pytorch.org/docs/stable/nested.html) allow for one of the tensor dimensions to be ragged, meaning that we can skip padding samples of different length and they get packed compactly in a tensor ([see here](https://docs.pytorch.org/docs/stable/nested.html#data-layout-and-shape)). NJTs need special support from operations to accommodate for their structure and `flex_attention` has added this support recently ([see here](https://docs.pytorch.org/blog/flexattention-for-inference/#ragged-input-sequences-with-nested-jagged-tensors-njts)). Unfortunately after some testing it turns out that it breaks when used with `torch.compile` and this combination of `flex_attention` + NJTs + `torch.compile` is discouraged for now by PyTorch team, [see here](https://github.com/pytorch/pytorch/issues/154556#issuecomment-2945803620). 

Left a commented out version of a variant using NJTs in nanoGPT-flex, it works without `torch.compile` but is slow. I also couldn't fully get rid of padding, needed to use a normal tensor with `sequence_ids`, because of the tensor indexing which is used in `document_causal_mask` (e.g. `sequence_ids[b, q_idx]`) and which is not supported with NJTs. 

For NJTs addition also needs to be done a bit differently than the regular `c = a + b`
```
# jagged version of e = tok_emb + pos_emb

ev = tok_emb.values() + pos_emb.values()
e = torch.nested.nested_tensor_from_jagged(values=ev, offsets=tok_emb.offsets())
```
When `tok_emb` and `pos_emb` are NJTs their dims may look like this `[32, j4, 192]` and `[32, j5, 192]`, where `jx` indicates a jagged dimension. And because the NJTs were created separately (as a result of embedding `ids` and `pos_ids`, which share the same structure but are created separately), they got a different dimension indicator. So although the addition makes sense to perform, PyTorch doesn't check that. To bypass that, we can take a flat view of NJT which we access by doing `x.values()` and perform the elementwise addition. After the operation we construct a NJT from the result by passing a flat tensor from the result of addition and inform PyTorch of its jagged structure by passing offsets ourselves. And that's the recommended approach from [NJTs docs](https://docs.pytorch.org/docs/stable/nested.html#ragged-structure-incompatibility).