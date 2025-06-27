# nanoGPT-recommender

Adaptation of [nanoGPT model](https://github.com/karpathy/nanoGPT) to a task of Amazon product recommendation for [Data Science Club PJATK kaggle challenge](https://www.kaggle.com/competitions/product-recommendation-challenge).

The idea is simple, let's treat purchases as sequences of tokens (item_ids), and based on user's current history let's predict next item that will be bought. During training we take available sequences and shift them one left to make targets (next word prediction target).
Additionally this repo leverages SigLIP embeddings to incorporate image data into the model.

Key changes to the original architecture:
- pack input sequences for more efficient training (70% of training dataset, ~600k rows, are sequences of length 2, so after making it a training example, it's just one input, one output token per row)
    - adapt causal attention mask, so independent samples don't influence each other, see [here](https://huggingface.co/blog/sirluk/llm-sequence-packing) for visual example of the attention masking that was needed
    - generate pos ids which start from 0 for each sample in a packed sequence, so positional info is reflected correctly during inference where samples will always have positional info starting from 0
- add SigLIP embedding layer
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