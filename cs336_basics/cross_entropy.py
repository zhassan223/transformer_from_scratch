import numpy as np
from .softmax import softmax
import torch
from torch import Tensor
def cross_entropy_loss(logits, targets):
    """
    Computes the cross entropy loss:
    li = -log softmax(oi)[xi+1]
    Args:
        logits: np.ndarray of shape (..., vocab_size)
        targets: np.ndarray of shape (...) with integer indices
    Returns:
        Average cross entropy loss over batch dimensions.
    """


    #he log-sum-exp form is mathematically equivalent to −log softmax but more robust for very large or small logits. 
    # which algebraically equals −log softmax(ls)[target] but never explicitly forms the division or two exponentials.
    logits_max=torch.max(logits,dim=-1,keepdim=True).values
    logits_stable=logits-logits_max

    log_sum_exp=torch.logsumexp(logits_stable,dim=-1)

    target_logit = torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = log_sum_exp - target_logit

    return loss.mean()

    