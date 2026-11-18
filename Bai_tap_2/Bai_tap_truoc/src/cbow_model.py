import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWFullSoftmax(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, context_ids):
        context_vectors = self.embeddings(context_ids)
        hidden = context_vectors.mean(dim=1)
        logits = self.output_layer(hidden)
        return logits


def test_one_hot_equivalence():
    torch.manual_seed(42)

    vocab_size = 7
    embedding_dim = 5

    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    W = embedding_layer.weight

    for index in range(vocab_size):
        one_hot = F.one_hot(
            torch.tensor(index),
            num_classes=vocab_size,
        ).to(dtype=W.dtype)

        vector_from_lookup = embedding_layer(
            torch.tensor(index)
        )

        vector_from_one_hot = one_hot @ W

        assert torch.allclose(
            vector_from_lookup,
            vector_from_one_hot,
            atol=1e-6,
        ), f"One-hot equivalence failed at token ID {index}"

    print("One-hot equivalence test: PASSED")


if __name__ == "__main__":
    test_one_hot_equivalence()

    model = CBOWFullSoftmax(
        vocab_size=7,
        embedding_dim=5,
    )

    context_ids = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ])

    logits = model(context_ids)

    assert logits.shape == (2, 7)

    print(f"Context shape: {context_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print("CBOW Full Softmax test: PASSED")
