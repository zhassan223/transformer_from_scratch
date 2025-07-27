import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        """
        Construct the RMSNorm module.

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.

        Args:
            x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        in_dtype=x.dtype
        x=x.to(torch.float32)

        squared_x = x**2
        #(1/d_model) * sum(x_i^2)
        mean_squared_x = torch.mean(squared_x,dim=-1,keepdim=True)
        rms_x=torch.sqrt(mean_squared_x+self.eps)

        normalized_x = x / (rms_x)

        output=self.weight * normalized_x

    
        return output.to(in_dtype)