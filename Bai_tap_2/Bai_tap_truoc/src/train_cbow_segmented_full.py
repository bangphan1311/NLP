import csv
import json
import math
import random
import time
from pathlib import Path

import torch

from cbow_negative_sampling_model import CBOWNegativeSampling


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/cbow/segmented_ids.jsonl"
VOCAB_PATH = ROOT / "data/processed/vocab/segmented_vocab.json"

BASE_CHECKPOINT = (
    ROOT / "models/cbow_segmented_negative_sampling_continued_10epochs.pt"
)
OUTPUT_CHECKPOINT = (
    ROOT / "models/cbow_segmented_negative_sampling_30epochs.pt"
)
LOSS_CSV = (
    ROOT / "results/cbow_segmented_negative_sampling_30epochs.csv"
)

EMBEDDING_DIM = 100
WINDOW_SIZE = 2
NUM_NEGATIVES = 5
BATCH_SIZE = 256
TARGET_EPOCHS = 30
SUBSAMPLE_THRESHOLD = 1e-5
SEED = 42


def load_vocab(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data.get("vocab") if isinstance(data, dict) else None
    frequencies = data.get("frequencies") if isinstance(data, dict) else None

    if not isinstance(vocab, dict) or not isinstance(frequencies, dict):
        raise ValueError("Vocabulary cần có 'vocab' và 'frequencies'.")

    if vocab.get("<unk>") != 0:
        raise ValueError("'<unk>' phải có ID bằng 0.")

    ids = list(vocab.values())

    if (
        not all(type(i) is int for i in ids)
        or len(set(ids)) != len(ids)
        or set(ids) != set(range(len(vocab)))
    ):
        raise ValueError("Token ID phải duy nhất và liên tục từ 0.")

    for token, token_id in vocab.items():
        if token_id != 0 and token not in frequencies:
            raise ValueError(f"Thiếu tần suất của token: {token}")

    return vocab, frequencies


def build_distributions(vocab, frequencies):
    total_frequency = sum(frequencies.values())

    if total_frequency <= 0:
        raise ValueError("Tổng tần suất không hợp lệ.")

    keep_probabilities = [1.0] * len(vocab)
    negative_probabilities = torch.zeros(
        len(vocab), dtype=torch.float32
    )

    for token, token_id in vocab.items():
        if token_id == 0:
            continue

        frequency = frequencies[token] / total_frequency

        keep_probabilities[token_id] = min(
            1.0,
            (
                math.sqrt(frequency / SUBSAMPLE_THRESHOLD) + 1.0
            ) * SUBSAMPLE_THRESHOLD / frequency,
        )

        negative_probabilities[token_id] = (
            float(frequencies[token]) ** 0.75
        )

    probability_sum = negative_probabilities.sum()

    if probability_sum <= 0:
        raise ValueError("Không thể tạo phân phối lấy mẫu âm.")

    negative_probabilities /= probability_sum

    return keep_probabilities, negative_probabilities


def get_document_offsets(path):
    offsets = []

    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()

            if not line:
                break

            if line.strip():
                offsets.append(offset)

    return offsets


def subsample_document(token_ids, keep_probabilities, rng):
    result = []

    for token_id in token_ids:
        if token_id == 0:
            result.append(0)
        elif rng.random() < keep_probabilities[token_id]:
            result.append(token_id)

    return result


def generate_samples(token_ids):
    for i in range(WINDOW_SIZE, len(token_ids) - WINDOW_SIZE):
        context = (
            token_ids[i - WINDOW_SIZE:i]
            + token_ids[i + 1:i + WINDOW_SIZE + 1]
        )
        target = token_ids[i]

        if target == 0 or any(token_id == 0 for token_id in context):
            continue

        yield context, target


def sample_negative_ids(targets, probabilities):
    batch_size = targets.shape[0]

    negatives = torch.multinomial(
        probabilities,
        num_samples=batch_size * NUM_NEGATIVES,
        replacement=True,
    ).view(batch_size, NUM_NEGATIVES)

    mask = negatives.eq(targets.unsqueeze(1))

    while mask.any():
        count = int(mask.sum().item())

        negatives[mask] = torch.multinomial(
            probabilities,
            num_samples=count,
            replacement=True,
        )

        mask = negatives.eq(targets.unsqueeze(1))

    return negatives


def train_batch(model, optimizer, contexts, targets, probabilities):
    x = torch.tensor(contexts, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    negative_ids = sample_negative_ids(y, probabilities)

    optimizer.zero_grad()
    loss = model(x, y, negative_ids)
    loss.backward()
    optimizer.step()

    return loss.item(), len(contexts)


def save_checkpoint(
    model,
    optimizer,
    vocab,
    epoch,
    losses,
    rng,
    original_config,
):
    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "epoch": epoch,
        "epoch_losses": losses,
        "embedding_dim": EMBEDDING_DIM,
        "vocab_size": len(vocab),
        "window_size": WINDOW_SIZE,
        "num_negatives": NUM_NEGATIVES,
        "config": original_config,
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "continued_to_epoch": TARGET_EPOCHS,
    }

    temp_path = OUTPUT_CHECKPOINT.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(OUTPUT_CHECKPOINT)


def save_losses(losses):
    LOSS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with LOSS_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "average_training_loss"])

        for epoch, loss in enumerate(losses, start=1):
            writer.writerow([epoch, loss])


def main():
    for path in (DATA_PATH, VOCAB_PATH, BASE_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    vocab, frequencies = load_vocab(VOCAB_PATH)
    keep_probabilities, negative_probabilities = build_distributions(
        vocab, frequencies
    )

    resume_path = (
        OUTPUT_CHECKPOINT
        if OUTPUT_CHECKPOINT.exists()
        else BASE_CHECKPOINT
    )

    checkpoint = torch.load(
        resume_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("vocab") != vocab:
        raise ValueError("Vocabulary trong checkpoint không khớp.")

    if checkpoint.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError("Embedding dimension không khớp.")

    if checkpoint.get("window_size") != WINDOW_SIZE:
        raise ValueError("Context window không khớp.")

    if checkpoint.get("num_negatives") != NUM_NEGATIVES:
        raise ValueError("Số negative samples không khớp.")

    original_config = checkpoint.get("config", {})
    expected_config = {
        "data_path": str(DATA_PATH),
        "vocab_path": str(VOCAB_PATH),
        "embedding_dim": EMBEDDING_DIM,
        "window_size": WINDOW_SIZE,
        "num_negatives": NUM_NEGATIVES,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.001,
        "subsample_threshold": SUBSAMPLE_THRESHOLD,
        "seed": SEED,
    }

    for key, expected_value in expected_config.items():
        if original_config.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint không khớp cấu hình tại '{key}'. "
                "Đã dừng để tránh tiếp tục sai mô hình."
            )

    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    if "python_rng_state" not in checkpoint:
        raise ValueError("Checkpoint thiếu trạng thái random.")

    if "torch_rng_state" not in checkpoint:
        raise ValueError("Checkpoint thiếu trạng thái PyTorch.")

    model = CBOWNegativeSampling(
        len(vocab),
        EMBEDDING_DIM,
        sparse_embeddings=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.SparseAdam(
        model.parameters(),
        lr=0.00004,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    losses = list(checkpoint["epoch_losses"])
    start_epoch = checkpoint["epoch"] + 1

    rng.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    if start_epoch > TARGET_EPOCHS:
        print(f"Mô hình đã hoàn thành epoch {checkpoint['epoch']}.")
        return

    offsets = get_document_offsets(DATA_PATH)

    print(f"Checkpoint đầu vào: {resume_path}")
    print(f"Đã hoàn thành: {checkpoint['epoch']} epoch")
    print(f"Sẽ tiếp tục: epoch {start_epoch} đến {TARGET_EPOCHS}")
    print(f"Corpus documents: {len(offsets):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    print("Optimizer: SparseAdam")
    print("Không khởi tạo lại trọng số mô hình.")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, TARGET_EPOCHS + 1):
        model.train()
        rng.shuffle(offsets)

        current_lr = 0.00004 * max(
            0.1,
            (TARGET_EPOCHS + 1 - epoch) / 20,
        )

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        contexts = []
        targets = []
        total_samples = 0
        total_loss = 0.0
        tokens_after_subsampling = 0

        with DATA_PATH.open("rb") as f:
            for offset in offsets:
                f.seek(offset)
                token_ids = json.loads(f.readline())

                token_ids = subsample_document(
                    token_ids,
                    keep_probabilities,
                    rng,
                )
                tokens_after_subsampling += len(token_ids)

                for context, target in generate_samples(token_ids):
                    contexts.append(context)
                    targets.append(target)

                    if len(contexts) >= BATCH_SIZE:
                        loss_value, count = train_batch(
                            model,
                            optimizer,
                            contexts,
                            targets,
                            negative_probabilities,
                        )

                        total_loss += loss_value * count
                        total_samples += count
                        contexts.clear()
                        targets.clear()

        if contexts:
            loss_value, count = train_batch(
                model,
                optimizer,
                contexts,
                targets,
                negative_probabilities,
            )

            total_loss += loss_value * count
            total_samples += count

        if total_samples == 0:
            raise RuntimeError("Không tạo được mẫu huấn luyện.")

        average_loss = total_loss / total_samples
        losses.append(average_loss)

        save_checkpoint(
            model,
            optimizer,
            vocab,
            epoch,
            losses,
            rng,
            original_config,
        )
        save_losses(losses)

        print(
            f"Epoch {epoch:02d}/{TARGET_EPOCHS} | "
            f"Samples: {total_samples:,} | "
            f"Tokens after subsampling: {tokens_after_subsampling:,} | "
            f"Loss: {average_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        print(f"Checkpoint saved: {OUTPUT_CHECKPOINT}")

    print("-" * 60)
    print(f"Loss đầu tiên: {losses[0]:.4f}")
    print(f"Loss sau 30 epoch: {losses[-1]:.4f}")
    print(
        f"Thời gian tiếp tục: "
        f"{(time.time() - start_time) / 60:.2f} phút"
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
