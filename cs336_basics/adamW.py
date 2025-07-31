import torch
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Optional
class AdamW(Optimizer):
    """
    Implements the AdamW optimizer (Adam with decoupled weight decay)

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate α.
        betas (Tuple[float, float], optional): Coefficients (β1, β2) used for
            computing running averages of gradient and its square. Default: (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical
            stability. Default: 1e-8.
        weight_decay (float, optional): Weight decay coefficient λ (decoupled).
            Default: 0.01.

    """
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        super().__init__(params, {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        })

        # state (per-parameter) will hold:
        #   step: int
        #   exp_avg: Tensor (first moment)
        #   exp_avg_sq: Tensor (second moment)
        # …  
        # Implementation of step()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr=group['lr']
            beta1,beta2=group['betas']
            eps=group['eps']
            wd=group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad.data
                state = self.state[p]
                if len(state)==0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["second_moment"] = torch.zeros_like(p.data)
                
                
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # Update biased first moment estimate
                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                denom = second_moment.sqrt().add_(eps)
                
                # Parameter update
                p.data.addcdiv_(first_moment, denom, value=-step_size)

        return loss
    
