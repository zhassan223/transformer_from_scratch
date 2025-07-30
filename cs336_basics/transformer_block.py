import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .rmsnorm import RMSNorm
from .multihead_self_attention import CausalMultiHeadSelfAttention
from .RotaryPositionsEmbedding import RotaryPositionalEmbedding
from .positionwise_ffn import PositionwiseFFN



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048,rope_theta=10000.0):
        super().__init__()
        
        # FIX: Define two separate RMSNorm layers with the correct names
        self.attn_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)

        rope = RotaryPositionalEmbedding(
            theta=rope_theta, 
            d_k=d_model // num_heads, 
            max_seq_len=max_seq_len
        )
        self.mha = CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope)
        self.ffn = PositionwiseFFN(d_model, d_ff)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        batch_size, seq_len, _ = x.shape

        # Generate token_positions dynamically inside the forward pass
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        attn_input = self.attn_norm(x)
        attn_output = self.mha(attn_input, token_positions)
        x = x + attn_output

        # Second residual connection (Feed-Forward)
        ffn_input = self.ff_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        
        return x