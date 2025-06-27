import torch

def create_packed_attention_mask(sequence_ids):
    """
    Create attention mask for packed sequences.
    
    Args:
        sequence_ids: (B, T) tensor where each element indicates which sequence 
                     the token belongs to. Different sequences should have different IDs.
    Returns:
        attention_mask: (B, T, T) tensor where attention_mask[b, i, j] = True if 
                       token i can attend to token j, False otherwise
    """
    B, T = sequence_ids.shape
    device = sequence_ids.device
    
    # Create a mask where tokens can only attend to tokens from the same sequence
    # sequence_ids: (B, T) -> (B, T, 1) and (B, 1, T)
    seq_mask = (sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(1))  # (B, T, T)
    
    # Create causal mask
    causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))  # (T, T)
    
    # Combine: can attend if same sequence AND causal
    attention_mask = seq_mask & causal_mask.unsqueeze(0)  # (B, T, T)
    
    return attention_mask.unsqueeze(1)
