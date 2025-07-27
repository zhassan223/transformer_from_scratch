import torch
from torch import nn
from torch.nn import functional as F

from cs336_basics.Linear import Linear

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        """
        Constructs a Position-wise Feed-Forward Network using the SwiGLU variant.

        This network consists of three linear transformations and a SiLU activation function,
        implementing the formula: FFN(x) = (SiLU(x @ W1) * (x @ W3)) @ W2.

        Args:
            d_model (int): The dimensionality of the input and output features.
            d_ff (int): The dimensionality of the inner hidden layer.
            device (torch.device | None, optional): The device to store parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): The data type for the parameters. Defaults to None.
        """
        super().__init__()
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU feed-forward network to the input tensor.
        Formula: (SiLU(x @ W1) * (x @ W3)) @ W2

        Args:
            x (torch.Tensor): The input tensor, with shape (..., d_model).

        Returns:
            torch.Tensor: The output tensor, with the same shape as the input (..., d_model).
        """
        # F.silu is numerically stable and efficient. It's equivalent to x * sigmoid(x).
        activated_path = F.silu(self.w1(x))
        
        gate_values = self.w3(x)
        
        gated_hidden_state = activated_path * gate_values
        
        return self.w2(gated_hidden_state)


