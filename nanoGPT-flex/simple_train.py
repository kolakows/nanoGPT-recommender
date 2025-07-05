import os
import ast
import math
import time
import joblib
import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
from model import GPT, GPTConfig
from train_utils import ItemSeqDataset, validate, SimpleDS, simpleds_collate_fn, collate_fn
from torch.utils.data import DataLoader

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)


train_df = pd.read_csv("train_df.csv")
train_df['item_id_enc'] = train_df['item_id_enc'].apply(ast.literal_eval)
test_df = pd.read_csv("test_df.csv")
test_df['item_id_enc'] = test_df['item_id_enc'].apply(ast.literal_eval)

train_df = pd.concat([train_df, test_df]).reset_index(drop=True)

le = joblib.load("label_encoder.joblib")

vocab_size = len(le.classes_) + 1
n_layer = 6
n_head = 6
n_embd = 192
block_size = 256
eval_interval = 100
ckpt_name = f"all_data_block_size_{block_size}"                       
ckpt_interval = 5_000
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda"

dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)


# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10_000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 10_000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

batch_size = 32
train_ds = ItemSeqDataset(train_df['item_id_enc'], block_size)
train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)
test_ds = ItemSeqDataset(test_df['item_id_enc'], block_size)
test_dl = DataLoader(test_ds, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)

valid_ds = SimpleDS(test_df['item_id_enc'])
valid_dl = DataLoader(valid_ds, batch_size=32, pin_memory=True, collate_fn=simpleds_collate_fn)

train_iter = iter(train_dl)
test_iter = iter(test_dl)

iterators = {}

def get_batch(split):
    dl = train_dl if split == 'train' else test_dl
    
    # Initialize iterator if it doesn't exista
    if split not in iterators:
        iterators[split] = iter(dl)
    
    try:
        x, y, seq, pos = next(iterators[split])
    except StopIteration:
        iterators[split] = iter(dl)
        x, y, seq, pos = next(iterators[split])
    
    return (x.to(device, non_blocking=True), 
            y.to(device, non_blocking=True), 
            seq.to(device, non_blocking=True),
            pos.to(device, non_blocking=True))

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Initialize model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size, dropout=0.0)
siglip_emb = torch.load("siglip_ordered_embedding.pth", weights_only=False)
model = GPT(GPTConfig(**model_args), siglip_emb)
model.to(device)
model.train()

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

model = torch.compile(model)

# Simple evaluation
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        loss_sum = 0
        for _ in range(10):  # Just 10 batches for quick eval
            X, Y, seq, pos = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, seq, pos)
            loss_sum += loss.item()
        losses[split] = loss_sum / 10
    model.train()
    return losses

t0 = time.time()

# Forward pass
X, Y, seq, pos = get_batch('train')

for iter_num in range(max_iters):

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Eval and logging
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        t1 = time.time()
        dt = t1 - t0
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {dt*1000:.2f}ms")
        t0 = t1
        

    # if iter_num % (5*eval_interval) == 0:
    #     validate(model, valid_dl, iter_num, ctx)

    if (iter_num+1) % ckpt_interval == 0:
        torch.save(model.state_dict(), f"{ckpt_name}_{iter_num}.pth")

    # Forward pass
    with ctx:
        logits, loss = model(X, Y, seq, pos)
    
    X, Y, seq, pos = get_batch('train')
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete!")