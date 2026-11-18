import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWHierarchicalSoftmax(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        frequencies_by_id: dict[int, int],
    ):
        super().__init__()

        if vocab_size < 3:
            raise ValueError(
                "Vocabulary cần <unk> và ít nhất hai token thường."
            )

        for token_id in range(1, vocab_size):
            frequency = frequencies_by_id.get(token_id)

            if (
                not isinstance(frequency, (int, float))
                or frequency <= 0
            ):
                raise ValueError(
                    f"Tần suất không hợp lệ cho token ID {token_id}."
                )

        paths, codes, path_lengths, num_internal_nodes = (
            self._build_huffman_tree(vocab_size, frequencies_by_id)
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_internal_nodes = num_internal_nodes

        self.input_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.output_embeddings = nn.Embedding(
            num_internal_nodes,
            embedding_dim,
        )

        bound = 0.5 / embedding_dim

        nn.init.uniform_(
            self.input_embeddings.weight,
            -bound,
            bound,
        )
        nn.init.uniform_(
            self.output_embeddings.weight,
            -bound,
            bound,
        )

        self.register_buffer("path_node_ids", paths)
        self.register_buffer("path_codes", codes)
        self.register_buffer("path_lengths", path_lengths)

    @staticmethod
    def _build_huffman_tree(vocab_size, frequencies_by_id):
        heap = []
        serial = 0

        for token_id in range(1, vocab_size):
            frequency = int(frequencies_by_id[token_id])
            leaf = ("leaf", token_id)

            heapq.heappush(
                heap,
                (frequency, serial, leaf),
            )
            serial += 1

        next_internal_id = 0

        while len(heap) > 1:
            left_frequency, _, left_node = heapq.heappop(heap)
            right_frequency, _, right_node = heapq.heappop(heap)

            internal_node = (
                "internal",
                next_internal_id,
                left_node,
                right_node,
            )

            heapq.heappush(
                heap,
                (
                    left_frequency + right_frequency,
                    serial,
                    internal_node,
                ),
            )

            serial += 1
            next_internal_id += 1

        root = heap[0][2]
        token_paths = {}

        def traverse(node, node_path, code_path):
            if node[0] == "leaf":
                token_id = node[1]
                token_paths[token_id] = (
                    node_path.copy(),
                    code_path.copy(),
                )
                return

            _, internal_id, left_node, right_node = node

            traverse(
                left_node,
                node_path + [internal_id],
                code_path + [0],
            )

            traverse(
                right_node,
                node_path + [internal_id],
                code_path + [1],
            )

        traverse(root, [], [])

        max_depth = max(
            len(token_paths[token_id][0])
            for token_id in range(1, vocab_size)
        )

        paths = torch.full(
            (vocab_size, max_depth),
            -1,
            dtype=torch.long,
        )

        codes = torch.zeros(
            (vocab_size, max_depth),
            dtype=torch.long,
        )

        path_lengths = torch.zeros(
            vocab_size,
            dtype=torch.long,
        )

        for token_id, (node_path, code_path) in token_paths.items():
            length = len(node_path)

            paths[token_id, :length] = torch.tensor(node_path)
            codes[token_id, :length] = torch.tensor(code_path)
            path_lengths[token_id] = length

        return paths, codes, path_lengths, next_internal_id

    def forward(self, context_ids, target_ids):
        if context_ids.ndim != 2:
            raise ValueError(
                "context_ids phải có shape [batch_size, context_size]."
            )

        if target_ids.ndim != 1:
            raise ValueError(
                "target_ids phải có shape [batch_size]."
            )

        if context_ids.shape[0] != target_ids.shape[0]:
            raise ValueError(
                "Batch context và target không khớp."
            )

        if torch.any(target_ids <= 0) or torch.any(
            target_ids >= self.vocab_size
        ):
            raise ValueError(
                "Target phải thuộc vocabulary và không được là <unk>."
            )

        context_vectors = self.input_embeddings(context_ids)
        hidden = context_vectors.mean(dim=1)

        node_ids = self.path_node_ids[target_ids]
        codes = self.path_codes[target_ids]
        lengths = self.path_lengths[target_ids]

        valid_mask = (
            torch.arange(
                node_ids.shape[1],
                device=node_ids.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        node_vectors = self.output_embeddings(
            node_ids.clamp_min(0)
        )

        scores = (
            node_vectors * hidden.unsqueeze(1)
        ).sum(dim=2)

        step_losses = torch.where(
            codes == 1,
            F.softplus(-scores),
            F.softplus(scores),
        )

        step_losses = step_losses * valid_mask

        return step_losses.sum(dim=1).mean()

    @torch.no_grad()
    def probabilities(self, context_ids):
        if context_ids.ndim == 1:
            context_ids = context_ids.unsqueeze(0)

        if context_ids.ndim != 2:
            raise ValueError(
                "context_ids phải là vector hoặc ma trận token ID."
            )

        hidden = self.input_embeddings(context_ids).mean(dim=1)

        batch_size = hidden.shape[0]

        probabilities = torch.zeros(
            batch_size,
            self.vocab_size,
            device=hidden.device,
        )

        for token_id in range(1, self.vocab_size):
            length = int(self.path_lengths[token_id].item())

            node_ids = self.path_node_ids[token_id, :length]
            codes = self.path_codes[token_id, :length]

            node_vectors = self.output_embeddings(node_ids)

            scores = (
                hidden.unsqueeze(1) * node_vectors.unsqueeze(0)
            ).sum(dim=2)

            log_probabilities = torch.where(
                codes.unsqueeze(0) == 1,
                F.logsigmoid(scores),
                F.logsigmoid(-scores),
            ).sum(dim=1)

            probabilities[:, token_id] = log_probabilities.exp()

        return probabilities


if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 6
    embedding_dim = 8

    frequencies_by_id = {
        1: 50,
        2: 40,
        3: 30,
        4: 20,
        5: 10,
    }

    model = CBOWHierarchicalSoftmax(
        vocab_size,
        embedding_dim,
        frequencies_by_id,
    )

    test_context = torch.tensor([1, 2, 3, 4])

    probabilities = model.probabilities(test_context)
    probability_sum = probabilities[0, 1:].sum().item()

    print("=== Probability normalization test ===")
    print(f"Sum of target probabilities: {probability_sum:.8f}")

    assert abs(probability_sum - 1.0) < 1e-5

    depths = model.path_lengths[1:].float()

    weighted_average_depth = (
        depths
        * torch.tensor(
            [
                frequencies_by_id[token_id]
                for token_id in range(1, vocab_size)
            ],
            dtype=torch.float32,
        )
    ).sum() / sum(frequencies_by_id.values())

    print(f"Mean Huffman depth: {depths.mean().item():.2f}")
    print(
        "Frequency-weighted mean depth: "
        f"{weighted_average_depth.item():.2f}"
    )

    context_batch = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ])

    target_batch = torch.tensor([2, 4])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    optimizer.zero_grad()
    loss = model(context_batch, target_batch)
    loss.backward()

    assert model.input_embeddings.weight.grad is not None
    assert model.output_embeddings.weight.grad is not None

    optimizer.step()

    print(f"One-step training loss: {loss.item():.6f}")
    print("Gradient test: PASSED")
    print("Hierarchical Softmax tests: PASSED")
