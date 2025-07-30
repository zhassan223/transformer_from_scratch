import torch 
from cs336_basics.softmax import softmax
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.
    softmax(QtK/sqrt(dk))@V
    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, ..., seq_len, d_k).
        key (torch.Tensor): Key tensor of shape (batch_size, ..., seq_len, d_k).
        value (torch.Tensor): Value tensor of shape (batch_size, ..., seq_len, d_v).
        mask (torch.Tensor, optional): Boolean mask of shape (seq_len, seq_len). True for valid positions, False for masked positions.

    Returns:
        torch.Tensor:
    """
    #q=batch_size ....seq,d_k
    #k=bs....,seq_len,d_k
    #qi
    attention_scores= torch.einsum('...qd,...kd-> ...qk',query,key)
    attention_scores= attention_scores/torch.sqrt(torch.tensor(query.shape[-1]))
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask.bool(), -1e9)
    
    #seq_len X seq_len
    soft_maxed_val=softmax(attention_scores,dim=-1)

    return torch.einsum('...qk,...kv ->...qv',soft_maxed_val,value)
        

