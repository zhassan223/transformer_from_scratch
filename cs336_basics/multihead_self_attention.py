import torch
import torch.nn as nn
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.RotaryPositionsEmbedding import RotaryPositionalEmbedding
from cs336_basics.Linear import Linear
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,rope:RotaryPositionalEmbedding = None, token_positions = None ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Add any other necessary parameters or layers here
        self.d_k=d_model//num_heads

        self.q_proj=Linear(d_model,d_model)
        self.k_proj=Linear(d_model,d_model)
        self.v_proj=Linear(d_model,d_model)
        self.o_proj=Linear(d_model,d_model)
        self.rope=rope
        self.token_positions=token_positions



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size,seq_len,_=x.shape
        # Implement the forward pass for causal multi-head self-attention

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        # 2. Reshape for multi-head processing
        # (B, S, D) -> (B, S, H, d_k) -> (B, H, S, d_k) where H=num_heads
        
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)


        if self.rope is not None  and self.token_positions is not None:
            
            q=self.rope(q,self.token_positions)
            k=self.rope(k,self.token_positions)

        # 3. Create causal (look-ahead) mask
        # Mask is (S, S), True for positions to keep (j <= i)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 5. Concatenate heads and reshape back to (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.o_proj(attn_output)

     