import argparse
import csv
import json
import math
import random
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn

from cbow_model import CBOWFullSoftmax
from cbow_hierarchical_softmax_model import CBOWHierarchicalSoftmax
from cbow_negative_sampling_model import CBOWNegativeSampling


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data/processed"
VOCAB_DIR = DATA_DIR / "vocab"
CBOW_DIR = DATA_DIR / "cbow"

MODEL_DIR = ROOT / "models/compare"
RESULT_DIR = ROOT / "results/compare"

EMBEDDING_DIM = 100
WINDOW_SIZE = 2
BATCH_SIZE = 256
NUM_NEGATIVES = 5
SUBSAMPLE_THRESHOLD = 1e-5
LEARNING_RATE = 0.001
SEED = 42
BENCHMARK_BATCHES = 200
BENCHMARK_LIMIT_HOURS = 3.0

LOSS_NAMES = ("softmax", "hs", "ns")
LOSS_LABELS = {
    "softmax": "Naive Softmax",
    "hs": "Hierarchical Softmax",
    "ns": "Negative Sampling",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and benchmark three CBOW loss functions."
    )

    parser.add_argument(
        "--loss",
        choices=LOSS_NAMES,
        default=None,
        help="Loss function to train. Not required for --benchmark.",
    )
    parser.add_argument(
        "--corpus",
        choices=("segmented", "unsegmented"),
        default="segmented",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark 200 batches for all three losses.",
    )

    args = parser.parse_args()

    if not args.benchmark and args.loss is None:
        parser.error("--loss is required unless --benchmark is used.")

    if args.num_docs < 1:
        parser.error("--num-docs must be positive.")

    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    return args


def corpus_paths(corpus_name):
    if corpus_name == "segmented":
        return (
            CBOW_DIR / "segmented_ids.jsonl",
            VOCAB_DIR / "segmented_vocab.json",
        )

    return (
        CBOW_DIR / "unsegmented_ids.jsonl",
        VOCAB_DIR / "unsegmented_vocab.json",
    )


def load_vocabulary(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Vocabulary không hợp lệ: {path}")

    vocab = data.get("vocab")
    frequencies = data.get("frequencies")

    if not isinstance(vocab, dict) or not isinstance(frequencies, dict):
        raise ValueError(
            f"{path} phải có hai trường 'vocab' và 'frequencies'."
        )

    if vocab.get("<unk>") != 0:
        raise ValueError("Token <unk> phải có ID bằng 0.")

    ids = list(vocab.values())

    if (
        not all(type(token_id) is int for token_id in ids)
        or len(set(ids)) != len(ids)
        or set(ids) != set(range(len(vocab)))
    ):
        raise ValueError("Token ID phải là số nguyên, duy nhất và liên tục.")

    frequencies_by_id = {}

    for token, token_id in vocab.items():
        if token_id == 0:
            continue

        frequency = frequencies.get(token)

        if (
            not isinstance(frequency, (int, float))
            or frequency <= 0
        ):
            raise ValueError(
                f"Thiếu tần suất hợp lệ cho token {token!r}."
            )

        frequencies_by_id[token_id] = int(frequency)

    return vocab, frequencies, frequencies_by_id


def load_document_offsets(path, num_docs):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy corpus: {path}")

    offsets = []

    with path.open("rb") as f:
        while len(offsets) < num_docs:
            offset = f.tell()
            line = f.readline()

            if not line:
                break

            if line.strip():
                offsets.append(offset)

    if not offsets:
        raise ValueError(f"Corpus không có document: {path}")

    if len(offsets) < num_docs:
        print(
            f"Warning: corpus chỉ có {len(offsets):,} "
            f"document, ít hơn N={num_docs:,}."
        )

    return offsets


def build_sampling_distributions(vocab, frequencies, vocab_size):
    total_frequency = sum(frequencies.values())

    if total_frequency <= 0:
        raise ValueError("Tổng tần suất vocabulary không hợp lệ.")

    keep_probabilities = [1.0] * vocab_size
    negative_probabilities = torch.zeros(
        vocab_size,
        dtype=torch.float32,
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

    total_weight = negative_probabilities.sum()

    if total_weight <= 0:
        raise ValueError("Không tạo được phân phối lấy mẫu âm.")

    negative_probabilities /= total_weight

    return keep_probabilities, negative_probabilities


def subsample_document(token_ids, keep_probabilities, rng):
    retained = []

    for token_id in token_ids:
        if token_id == 0:
            retained.append(0)
        elif rng.random() < keep_probabilities[token_id]:
            retained.append(token_id)

    return retained


def generate_samples(token_ids):
    for index in range(WINDOW_SIZE, len(token_ids) - WINDOW_SIZE):
        context = (
            token_ids[index - WINDOW_SIZE:index]
            + token_ids[index + 1:index + WINDOW_SIZE + 1]
        )
        target = token_ids[index]

        if target == 0 or any(token_id == 0 for token_id in context):
            continue

        yield context, target


def epoch_batches(
    corpus_path,
    offsets,
    keep_probabilities,
    seed,
    epoch,
):
    rng = random.Random(seed + epoch)
    ordered_offsets = list(offsets)
    rng.shuffle(ordered_offsets)

    contexts = []
    targets = []

    with corpus_path.open("rb") as f:
        for offset in ordered_offsets:
            f.seek(offset)
            token_ids = json.loads(f.readline())

            token_ids = subsample_document(
                token_ids,
                keep_probabilities,
                rng,
            )

            for context, target in generate_samples(token_ids):
                contexts.append(context)
                targets.append(target)

                if len(contexts) == BATCH_SIZE:
                    yield contexts, targets
                    contexts = []
                    targets = []

    if contexts:
        yield contexts, targets


def count_epoch_samples(
    corpus_path,
    offsets,
    keep_probabilities,
    seed,
    epoch,
):
    total = 0
    rng = random.Random(seed + epoch)
    ordered_offsets = list(offsets)
    rng.shuffle(ordered_offsets)

    with corpus_path.open("rb") as f:
        for offset in ordered_offsets:
            f.seek(offset)
            token_ids = json.loads(f.readline())

            token_ids = subsample_document(
                token_ids,
                keep_probabilities,
                rng,
            )

            total += sum(1 for _ in generate_samples(token_ids))

    return total


def create_model(loss_name, vocab_size, frequencies_by_id):
    if loss_name == "softmax":
        return CBOWFullSoftmax(vocab_size, EMBEDDING_DIM)

    if loss_name == "hs":
        return CBOWHierarchicalSoftmax(
            vocab_size,
            EMBEDDING_DIM,
            frequencies_by_id,
        )

    return CBOWNegativeSampling(
        vocab_size,
        EMBEDDING_DIM,
        sparse_embeddings=False,
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def sample_negative_ids(targets, probabilities, generator):
    batch_size = targets.shape[0]

    negatives = torch.multinomial(
        probabilities,
        num_samples=batch_size * NUM_NEGATIVES,
        replacement=True,
        generator=generator,
    ).view(batch_size, NUM_NEGATIVES)

    mask = negatives.eq(targets.unsqueeze(1))

    while mask.any():
        count = int(mask.sum().item())

        negatives[mask] = torch.multinomial(
            probabilities,
            num_samples=count,
            replacement=True,
            generator=generator,
        )

        mask = negatives.eq(targets.unsqueeze(1))

    return negatives


def calculate_loss(
    model,
    loss_name,
    contexts,
    targets,
    negative_probabilities,
    negative_generator,
    criterion,
):
    x = torch.tensor(contexts, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    if loss_name == "softmax":
        logits = model(x)
        return criterion(logits, y)

    if loss_name == "hs":
        return model(x, y)

    negative_ids = sample_negative_ids(
        y,
        negative_probabilities,
        negative_generator,
    )

    return model(x, y, negative_ids)


def create_config(corpus_name, num_docs, epochs):
    return {
        "corpus": corpus_name,
        "num_docs": num_docs,
        "epochs": epochs,
        "embedding_dim": EMBEDDING_DIM,
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "num_negatives": NUM_NEGATIVES,
        "subsample_threshold": SUBSAMPLE_THRESHOLD,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "optimizer": "Adam",
        "learning_rate_schedule": "constant",
    }


def checkpoint_paths(corpus_name, loss_name):
    return (
        MODEL_DIR / f"{corpus_name}_{loss_name}.pt",
        RESULT_DIR / f"{corpus_name}_{loss_name}.csv",
    )


def save_checkpoint(path, model, optimizer, vocab, config, epoch, losses):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "config": config,
        "epoch": epoch,
        "epoch_losses": losses,
    }

    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def save_loss_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "seconds"])
        writer.writerows(rows)


def benchmark_model(
    loss_name,
    vocab_size,
    frequencies_by_id,
    negative_probabilities,
    benchmark_batches,
):
    torch.manual_seed(SEED)

    model = create_model(
        loss_name,
        vocab_size,
        frequencies_by_id,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()
    negative_generator = torch.Generator(device="cpu")
    negative_generator.manual_seed(SEED + 10000)

    model.train()
    elapsed = 0.0

    start = time.perf_counter()

    for contexts, targets in benchmark_batches:
        optimizer.zero_grad()

        loss = calculate_loss(
            model,
            loss_name,
            contexts,
            targets,
            negative_probabilities,
            negative_generator,
            criterion,
        )

        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start

    return {
        "loss": loss_name,
        "seconds_per_batch": elapsed / len(benchmark_batches),
        "benchmark_seconds": elapsed,
        "parameters": count_parameters(model),
    }


def run_benchmark(
    args,
    corpus_path,
    offsets,
    vocab,
    frequencies,
    frequencies_by_id,
):
    keep_probabilities, negative_probabilities = (
        build_sampling_distributions(
            vocab,
            frequencies,
            len(vocab),
        )
    )

    print("Preparing one shared sample sequence for benchmarking...")

    benchmark_batches = list(
        islice(
            epoch_batches(
                corpus_path,
                offsets,
                keep_probabilities,
                SEED,
                epoch=1,
            ),
            BENCHMARK_BATCHES,
        )
    )

    if len(benchmark_batches) < BENCHMARK_BATCHES:
        raise RuntimeError(
            f"Chỉ sinh được {len(benchmark_batches)} batch, "
            f"cần {BENCHMARK_BATCHES}."
        )

    total_samples = count_epoch_samples(
        corpus_path,
        offsets,
        keep_probabilities,
        SEED,
        epoch=1,
    )

    steps_per_epoch = math.ceil(total_samples / BATCH_SIZE)

    print(f"Corpus: {args.corpus}")
    print(f"Documents: {len(offsets):,}")
    print(f"Samples per estimated epoch: {total_samples:,}")
    print(f"Batches per epoch: {steps_per_epoch:,}")
    print(f"Benchmark batches per loss: {BENCHMARK_BATCHES}")
    print("-" * 65)

    results = []

    for loss_name in LOSS_NAMES:
        print(f"Benchmarking {LOSS_LABELS[loss_name]}...")

        result = benchmark_model(
            loss_name,
            len(vocab),
            frequencies_by_id,
            negative_probabilities,
            benchmark_batches,
        )

        result["estimated_seconds_per_epoch"] = (
            result["seconds_per_batch"] * steps_per_epoch
        )

        result["estimated_hours_for_five_epochs"] = (
            result["estimated_seconds_per_epoch"] * 5 / 3600
        )

        results.append(result)

        print(
            f"{LOSS_LABELS[loss_name]}: "
            f"{result['seconds_per_batch']:.4f} sec/batch | "
            f"{result['estimated_seconds_per_epoch'] / 60:.2f} min/epoch | "
            f"{result['estimated_hours_for_five_epochs']:.2f} hours/5 epochs | "
            f"{result['parameters']:,} parameters"
        )

    softmax_result = next(
        result for result in results if result["loss"] == "softmax"
    )

    estimated_full_hours = (
        softmax_result["estimated_hours_for_five_epochs"]
        * len(offsets) / args.num_docs
    )

    print("-" * 65)
    print(
        "Naive Softmax estimated time for the full corpus "
        f"and 5 epochs: {estimated_full_hours:.2f} hours"
    )

    if estimated_full_hours <= BENCHMARK_LIMIT_HOURS:
        recommended_docs = len(offsets)
        print(
            f"Recommendation: N = {recommended_docs:,} "
            "(full corpus; estimate within 3 hours)."
        )
    else:
        fraction = BENCHMARK_LIMIT_HOURS / estimated_full_hours
        recommended_docs = max(
            1000,
            int((len(offsets) * fraction) // 1000) * 1000,
        )
        recommended_docs = min(recommended_docs, len(offsets))

        print(
            f"Recommendation: N = {recommended_docs:,} "
            "to target about 3 hours for Softmax over 5 epochs."
        )

    print(
        "\nBenchmark complete. No model checkpoint was created or modified."
    )
    print(
        "Confirm N before launching training; do not run training yet."
    )


def train_model(
    args,
    corpus_path,
    offsets,
    vocab,
    frequencies,
    frequencies_by_id,
):
    model_path, csv_path = checkpoint_paths(
        args.corpus,
        args.loss,
    )

    config = create_config(
        args.corpus,
        len(offsets),
        args.epochs,
    )

    keep_probabilities, negative_probabilities = (
        build_sampling_distributions(
            vocab,
            frequencies,
            len(vocab),
        )
    )

    torch.manual_seed(SEED)

    model = create_model(
        args.loss,
        len(vocab),
        frequencies_by_id,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    losses = []
    rows = []
    start_epoch = 1

    if model_path.exists():
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("config") != config:
            raise ValueError(
                f"Checkpoint {model_path} có cấu hình khác. "
                "Đã dừng để không ghi đè hoặc tiếp tục nhầm."
            )

        if checkpoint.get("vocab") != vocab:
            raise ValueError("Vocabulary checkpoint không khớp.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        losses = list(checkpoint["epoch_losses"])
        start_epoch = checkpoint["epoch"] + 1

        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))[1:]

    else:
        if csv_path.exists():
            raise ValueError(
                f"CSV đã tồn tại nhưng checkpoint chưa có: {csv_path}. "
                "Không ghi đè kết quả cũ."
            )

    if start_epoch > args.epochs:
        print(
            f"{LOSS_LABELS[args.loss]} đã hoàn thành "
            f"{checkpoint['epoch']} epoch."
        )
        return

    print(f"Loss: {LOSS_LABELS[args.loss]}")
    print(f"Corpus: {args.corpus}")
    print(f"Documents: {len(offsets):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Training epochs: {start_epoch}–{args.epochs}")
    print(f"Optimizer: Adam, lr={LEARNING_RATE}")
    print(f"Output checkpoint: {model_path}")
    print("-" * 65)

    criterion = nn.CrossEntropyLoss()
    training_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        negative_generator = torch.Generator(device="cpu")
        negative_generator.manual_seed(SEED + 10000 + epoch)

        total_loss = 0.0
        total_samples = 0
        epoch_start = time.perf_counter()

        for contexts, targets in epoch_batches(
            corpus_path,
            offsets,
            keep_probabilities,
            SEED,
            epoch,
        ):
            optimizer.zero_grad()

            loss = calculate_loss(
                model,
                args.loss,
                contexts,
                targets,
                negative_probabilities,
                negative_generator,
                criterion,
            )

            loss.backward()
            optimizer.step()

            batch_size = len(contexts)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("Không tạo được mẫu huấn luyện.")

        average_loss = total_loss / total_samples
        epoch_seconds = time.perf_counter() - epoch_start

        losses.append(average_loss)
        rows.append([
            epoch,
            f"{average_loss:.8f}",
            f"{epoch_seconds:.4f}",
        ])

        save_checkpoint(
            model_path,
            model,
            optimizer,
            vocab,
            config,
            epoch,
            losses,
        )

        save_loss_csv(csv_path, rows)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Samples: {total_samples:,} | "
            f"Loss: {average_loss:.6f} | "
            f"Seconds: {epoch_seconds:.2f}"
        )
        print(f"Checkpoint saved: {model_path}")

    print("-" * 65)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(
        f"Training time: "
        f"{(time.perf_counter() - training_start) / 60:.2f} minutes"
    )
    print(f"CSV saved: {csv_path}")


def main():
    args = parse_args()

    torch.set_num_threads(
        max(1, torch.get_num_threads())
    )

    corpus_path, vocab_path = corpus_paths(args.corpus)

    vocab, frequencies, frequencies_by_id = load_vocabulary(
        vocab_path
    )

    offsets = load_document_offsets(
        corpus_path,
        args.num_docs,
    )

    print(f"PyTorch CPU threads: {torch.get_num_threads()}")

    if args.benchmark:
        run_benchmark(
            args,
            corpus_path,
            offsets,
            vocab,
            frequencies,
            frequencies_by_id,
        )
        return

    train_model(
        args,
        corpus_path,
        offsets,
        vocab,
        frequencies,
        frequencies_by_id,
    )


if __name__ == "__main__":
    main()
