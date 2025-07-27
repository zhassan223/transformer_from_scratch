import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta (float): Θ value for the RoPE.
            d_k (int): Dimension of query and key vectors.
            max_seq_len (int): Maximum sequence length that will be inputted.
            device (torch.device | None): Device to store the buffer on.
        """
        super().__init__()
        self.d_k=d_k
        self.theta= theta 
        self.max_seq_len = max_seq_len


        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        # Outer product: (max_seq_len, d_k//2
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # (max_seq_len, d_k//2)
        # Precompute cos and sin
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) specifying token positions.

        Returns:
            torch.Tensor: Tensor of the same shape as x with RoPE
        """
        original_shape=x.shape
        *batch_dims,seq_len,d_k = x.shape



        assert d_k == self.d_k, " d_k must match"
        
        #make last dim into pairs
        x_=x.view(*batch_dims, seq_len,d_k//2,2)
        x1,x2=x_[...,0],x_[...,1]

        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]

        #rotation
        x_rotated_0=x1 * cos - x2 *sin 
        x_rotated_1=x1*sin +  x2 * cos

        x_rotated_stacked=torch.stack([x_rotated_0,x_rotated_1],dim=-1)
        x_rotated_stacked=x_rotated_stacked.view(*batch_dims,seq_len,d_k)

        return x_rotated_stacked
