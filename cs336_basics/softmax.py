import torch
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies the softmax operation to the specified dimension of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension along which to apply softmax.

    Returns:
        torch.Tensor: The tensor with softmax applied along the specified
    """
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x_exp=torch.exp(x-x_max)
    x_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    return x_exp/x_sum