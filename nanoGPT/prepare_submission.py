import json
import torch
import joblib
import pandas as pd
from tqdm import tqdm
from model import GPT, GPTConfig

ratings = pd.read_csv('../data/train.csv')
item_metadata = pd.read_csv('../data/item_metadata.csv')
test_users = pd.read_csv('../data/test.csv')["user_id"]
mapping = json.load(open('../data/id_mappings.json'))

relevant_items = set(mapping['item_mapping'].keys())
item_metadata = item_metadata.query("parent_asin in @relevant_items").reset_index(drop=True)
le = joblib.load("label_encoder.joblib")
ratings['item_id_enc'] = le.transform(ratings['item_id'])

sequences = ratings.groupby('user_id')['item_id_enc'].agg(list).reset_index()

user_to_items = {usr:items for usr, items in zip(sequences["user_id"], sequences["item_id_enc"])}
print("data loaded")


vocab_size = len(le.classes_) + 1
n_layer = 6
n_head = 6
n_embd = 192
block_size = 32
device = "cuda"
ckpt_name = "all_data_19999.pth"

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size, dropout=0.0)
siglip_emb = torch.load("siglip_ordered_embedding.pth", weights_only=False)
model = GPT(GPTConfig(**model_args), siglip_emb)

state_dict = torch.load(ckpt_name)
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

test_preds = []
test_preds_seq = []
block_size = 32
batch_size = 64

# Group users into batches
user_batches = [test_users[i:i + batch_size] for i in range(0, len(test_users), batch_size)]

for batch_users in tqdm(user_batches):
    # Prepare batch data
    batch_x = []
    seq_lengths = []
    for usr in batch_users:
        x = user_to_items[usr]
        x = torch.tensor(x[-block_size:], dtype=torch.long, device="cuda")
        original_len = len(x)
        seq_lengths.append(original_len)
        
        # Right pad if sequence is shorter than block_size
        if len(x) < block_size:
            padding = torch.zeros(block_size - len(x), dtype=torch.long, device="cuda")
            x = torch.cat([x, padding])  # Right padding
        batch_x.append(x)
    
    # Stack into batch tensor
    batch_x = torch.stack(batch_x)  # Shape: (batch_size, block_size)
    
    with ctx:
        logits, _ = model(batch_x)
    
    # Get top predictions for each user in the batch using correct positions
    batch_top_preds = []
    for i, seq_len in enumerate(seq_lengths):
        last_valid_pos = seq_len - 1  # Last valid position for this sequence
        user_logits = logits[i, last_valid_pos, :]  # Extract logits at last valid position
        vals, top_preds = torch.topk(user_logits, 10)
        batch_top_preds.append(top_preds)
    
    # Add results for each user
    for i, usr in enumerate(batch_users):
        test_preds.append((usr, batch_top_preds[i].cpu().numpy().flatten().tolist()))

test_preds_corrected = [(usr, le.inverse_transform(preds)) for usr, preds in tqdm(test_preds)]

final_preds = [(usr, " ".join(str(x) for x in p)) for usr, p in test_preds_corrected]
submission_df = pd.DataFrame(final_preds)
submission_df.columns = ["user_id", "predictions"]
submission_df.to_csv(f"submission_all_data_{ckpt_name}.csv", index=False)
