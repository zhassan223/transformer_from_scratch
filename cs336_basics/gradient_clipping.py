from collections.abc import Iterable
import torch

def clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clips the gradients of the given parameters in-place so that the total
    l2 norm of all gradients does not exceed `max_norm`.

    This uses the formula:
        clip_coef = max_norm / (total_norm + eps)
        if clip_coef < 1:
            grad = grad * clip_coef

    Args:
        parameters (Iterable[torch.nn.Parameter]):
            Iterable of model parameters whose .grad will be clipped.
        max_norm (float):
            Maximum allowed l2 norm of the gradients.
        eps (float):
            Small constant for numerical stability. Default: 1e-6.

    Returns:
        None
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    total_norm_sq = 0.0
    for g in grads:
        #l2 norm
        param_norm = g.data.norm(2)
        total_norm_sq += param_norm.item() ** 2
    #mag
    total_norm = total_norm_sq ** 0.5

   
    clip_coef = max_norm / (total_norm + eps)
    #don't want it to blow up
    if clip_coef < 1.0:
        for g in grads:
            g.data.mul_(clip_coef)


