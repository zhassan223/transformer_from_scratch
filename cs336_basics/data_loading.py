import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Samples `batch_size` sequences of length `context_length` from `dataset`
    (a 1D array of token IDs) and returns the input/target pairs.

    Args:
        dataset (np.ndarray): 1D array of token IDs, shape (n,).
        batch_size (int): Number of sequences to sample (B).
        context_length (int): Length of each input sequence (m).
        device (str): Torch device string (e.g., 'cpu' or 'cuda:0').

    Returns:
        inputs (LongTensor): shape (B, m), each row is a context.
        targets (LongTensor): shape (B, m), each row is the next-token sequence.
    """
    n = dataset.shape[0]
    # sample valid start positions so that i + context_length < n
    starts = np.random.randint(0, n - context_length, size=batch_size)
    xs = [dataset[i : i + context_length] for i in starts]
    ys = [dataset[i + 1 : i + context_length + 1] for i in starts]

    inputs = torch.tensor(xs, dtype=torch.long, device=device)
    targets = torch.tensor(ys, dtype=torch.long, device=device)
    return inputs, targets