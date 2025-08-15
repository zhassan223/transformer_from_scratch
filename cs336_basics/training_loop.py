import torch
import numpy as np
from cs336_basics.data_loading import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamW import AdamW
from cs336_basics.cross_entropy import cross_entropy_loss

def train(
    dataset_path: str,
    vocab_size: int,
    batch_size: int,
    context_length: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    checkpoint_path: str,
    device: str = "cuda:0",
):
    """
    Runs a training loop for the Transformer language model.

    Args:
        dataset_path (str): Path to the dataset (tokenized as a 1D array of token IDs).
        vocab_size (int): Size of the vocabulary.
        batch_size (int): Number of sequences per batch.
        context_length (int): Length of each input sequence.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay coefficient for AdamW.
        num_epochs (int): Number of epochs to train.
        checkpoint_path (str): Path to save/load checkpoints.
        device (str): PyTorch device (e.g., 'cpu' or 'cuda:0').
    """
    # Load dataset with np.memmap for memory efficiency
    dataset = np.memmap(dataset_path, dtype=np.int32, mode="r")

    # Initialize model, optimizer, and other components
    model = TransformerLM(vocab_size=vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load checkpoint if available
    try:
        iteration = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resumed training from iteration {iteration}.")
    except FileNotFoundError:
        iteration = 0
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(num_epochs):
        for step in range(len(dataset) // (batch_size * context_length)):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)

            # Forward pass
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

            # Save checkpoint periodically
            if step % 1000 == 0:
                save_checkpoint(model, optimizer, iteration + step, checkpoint_path)

        print(f"Epoch {epoch} completed.")

    print("Training complete.")