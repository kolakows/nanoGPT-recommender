import time
import torch
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from tqdm import tqdm


def pad_row(row, block_size, pad_token):
    pad_len = block_size - len(row)
    row.extend(pad_len*[pad_token])

def pack_tokens(sequences, block_size, pad_token):
    batch = []
    row = []
    for seq in sequences:
        if len(seq) + len(row) <= block_size:
            row.extend(seq)
        else:
            pad_row(row, block_size, pad_token)
            batch.append(row)
            row = [*seq[:block_size]]
    if row:
        pad_row(row, block_size, pad_token)
        batch.append(row)
    return batch

def collate_fn(batch):
    x, y, seq, pos = list(zip(*batch))
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    seq = torch.tensor(seq, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.long)
    return x, y, seq, pos


def create_position_ids(sequence_ids):
    """
    Create position IDs that reset for each new sequence.
    This ensures position embeddings don't leak across sequence boundaries.
    
    Args:
        sequence_ids: List of lists of sequence IDs
    
    Returns:
        List of lists of position IDs with same structure as input
    """
    position_ids = []
    
    for seq_list in sequence_ids:
        seq_position_ids = []
        current_pos = 0
        current_seq_id = seq_list[0] if seq_list else None
        
        for seq_id in seq_list:
            if seq_id != current_seq_id and seq_id != -1:
                # New sequence started (and not padding)
                current_pos = 0
                current_seq_id = seq_id
            
            if seq_id != -1:  # Not padding
                seq_position_ids.append(current_pos)
                current_pos += 1
            else:
                seq_position_ids.append(0)  # Padding tokens get position 0
        
        position_ids.append(seq_position_ids)
    
    return position_ids

class ItemSeqDataset(Dataset):

    target_pad = -100
    seq_pad = -1
    input_pad = 0

    def __init__(self, sequences, block_size):
        super().__init__()
        sequences = sequences.tolist()
        shuffle(sequences)

        X = [s[:-1] for s in sequences]
        Y = [s[1:] for s in sequences]
        seq_ids = [len(s)*[i] for i, s in enumerate(X, 1)]
        pos_ids = create_position_ids(seq_ids)

        assert len(X) == len(Y) == len(seq_ids)
        self.packed_x = pack_tokens(X, block_size, self.input_pad)
        self.packed_y = pack_tokens(Y, block_size, self.target_pad)
        self.packed_seq_ids = pack_tokens(seq_ids, block_size, self.seq_pad)
        self.packed_pos_ids = pack_tokens(pos_ids, block_size, self.input_pad)
    
    def __getitem__(self, index):
        x, y = self.packed_x[index], self.packed_y[index]
        seq_ids = self.packed_seq_ids[index]
        pos_ids = self.packed_pos_ids[index]
        return x, y, seq_ids, pos_ids
    
    def __len__(self):
        return len(self.packed_x)
    

def calculate_map_at_k_torch(pairs, k=10):
    """
    PyTorch-based MAP@K calculation
    
    Args:
        pairs: List of tuples (candidates_tensor, pred_tensor)
        k: Number of top items to consider
    
    Returns:
        MAP@K score as torch tensor
    """
    ap_scores = []
    
    for candidates, pred in pairs:
        
        # Take top k
        top_k = candidates[:k]
        
        # Find positions where prediction matches candidates
        matches = (top_k == pred).float()
        
        if matches.sum() > 0:
            # Get position of first match (0-indexed)
            pos = torch.argmax(matches).item()
            ap = 1.0 / (pos + 1)
        else:
            ap = 0.0
        
        ap_scores.append(ap)
    
    return torch.tensor(ap_scores).mean()


@torch.no_grad()
def validate(model, valid_dl, iter_num, ctx):
    val_preds = []
    block_size = 32
    t1 = time.time()
    model.eval()

    device = next(model.parameters()).device
    
    for x, y in tqdm(valid_dl):
        x = x[:, -block_size:].to(device)
        y = y.to(device)
        
        with ctx:
            logits, _ = model(x)
        
        # Find last non-padding position for each sequence
        last_non_pad_pos = (x != 0).sum(dim=1) - 1  # -1 for 0-indexing
        batch_indices = torch.arange(len(x), device=x.device)
        last_logits = logits[batch_indices, last_non_pad_pos]
        
        vals, top_preds = torch.topk(last_logits, 10)
        
        for i in range(len(y)):
            val_preds.append((top_preds[i].cpu(), y[i].cpu()))
    
    top_k_map = calculate_map_at_k_torch(val_preds)
    t2 = time.time()
    print(f"step {iter_num}: map top_k oneshot {top_k_map:.4f}, time clc: {t2-t1:.4f}")
    model.train()

class SimpleDS(Dataset):
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def __getitem__(self, index):
        seq = self.sequences[index]
        x, y = seq[:-1], seq[-1]
        return x, y
    
    def __len__(self):
        return len(self.sequences)
    

def simpleds_collate_fn(batch):
    pad_id = 0
    x, y = list(zip(*batch))
    max_seq_len = max([len(s) for s in x])
    x = [s + (max_seq_len-len(s)) * [pad_id] for s in x]
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return x, y