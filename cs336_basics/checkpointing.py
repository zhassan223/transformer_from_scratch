import torch
from typing import Union, BinaryIO, IO
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
) -> None:
    """
    Save the model, optimizer, and iteration state to a checkpoint file.
    """
    checkpoint={
        'model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'iteration':iteration
   }
    torch.save(checkpoint,out)

def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load the model, optimizer, and iteration state from a checkpoint file.
    Returns the saved iteration number.
    """
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]