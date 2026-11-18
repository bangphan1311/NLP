import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWNegativeSampling(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        sparse_embeddings: bool = False,
    ):
        super().__init__()

        self.input_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            sparse=sparse_embeddings,
        )

        self.output_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            sparse=sparse_embeddings,
        )

        bound = 0.5 / embedding_dim

        nn.init.uniform_(
            self.input_embeddings.weight,
            -bound,
            bound,
        )

        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, context_ids, target_ids, negative_ids):
        context_vectors = self.input_embeddings(context_ids)
        hidden = context_vectors.mean(dim=1)

        positive_vectors = self.output_embeddings(target_ids)
        positive_scores = (hidden * positive_vectors).sum(dim=1)

        negative_vectors = self.output_embeddings(negative_ids)
        negative_scores = torch.bmm(
            negative_vectors,
            hidden.unsqueeze(2),
        ).squeeze(2)

        positive_loss = F.softplus(-positive_scores)
        negative_loss = F.softplus(negative_scores).sum(dim=1)

        return (positive_loss + negative_loss).mean()
