import torch
from torch import nn
from torch import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """
        Construct a linear transformation module.

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__() #
        self.W = nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=2/(in_features+out_features), a=-3.0, b=3.0) 




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

        Args:
            x: torch.Tensor The input tensor.

        Returns:
            torch.Tensor: The output tensor after

        """
        return einsum('...i,io->...o' , x, self.W)
    
