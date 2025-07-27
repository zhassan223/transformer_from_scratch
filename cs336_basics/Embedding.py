import torch
from torch import nn
from torch.nn.init import trunc_normal_

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """
        Construct an embedding module.

        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__() # Call the superclass constructor
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device, dtype=dtype))
        trunc_normal_(self.embedding_matrix, mean=0.0, std=1,a=-3.0,b=3.0)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: torch.Tensor Token IDs to lookup embeddings for

        Returns:
            torch.Tensor: The embedding vectors for the given
        """
        return self.embedding_matrix[token_ids]