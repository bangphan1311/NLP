import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_words import CONTROL_PAIRS, MIN_FREQ, RANDOM_WORDS, RELATED_PAIRS

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data/processed/vocab/segmented_vocab.json"
OUT_DIR = ROOT / "results/compare"
TOP_K = 5

MODELS = [
    ("softmax", ROOT / "models/compare/segmented_softmax.pt",
     ROOT / "results/compare/segmented_softmax.csv", 13_161_078),
    ("hs", ROOT / "models/compare/segmented_hs.pt",
     ROOT / "results/compare/segmented_hs.csv", 13_095_400),
    ("ns", ROOT / "models/compare/segmented_ns.pt",
     ROOT / "results/compare/segmented_ns.csv", 13_095_600),
    ("main_ns_30ep", ROOT / "models/cbow_segmented_negative_sampling_30epochs.pt",
     ROOT / "results/cbow_segmented_negative_sampling_30epochs.csv", 13_095_600),
]


def load_vocab():
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = data["vocab"]
    freq = torch.zeros(len(vocab))
    for token, token_id in vocab.items():
        if token_id != 0:
            freq[token_id] = float(data["frequencies"][token])
    return vocab, freq


def find_input_embedding(state_dict, vocab_size):
    candidates = [
        (k, v) for k, v in state_dict.items()
        if v.dim() == 2 and v.shape[0] == vocab_size
    ]

    def score(name):
        n = name.lower()
        if "input" in n:
            return 0
        if "embed" in n and "output" not in n and "node" not in n:
            return 1
        return 2

    candidates.sort(key=lambda kv: score(kv[0]))
    if not candidates or score(candidates[0][0]) == 2:
        raise ValueError(
            "Không xác định được input embedding. Ứng viên: "
            f"{[(k, tuple(v.shape)) for k, v in candidates]}"
        )
    print(f"   input embedding = '{candidates[0][0]}' (ứng viên: {[k for k, _ in candidates]})")
    return candidates[0][1].float()


def read_losses(path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []
    key = "loss" if "loss" in rows[0] else "average_training_loss"
    losses = [float(r[key]) for r in rows]
    seconds = [float(r["seconds"]) for r in rows] if "seconds" in rows[0] else []
    return losses, seconds


def geometry(M, k=2000):
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(M.shape[0], generator=g)[:k]
    X = F.normalize(M[idx], dim=1)
    C = X @ X.T
    m = C.shape[0]
    off_diag_mean = (C.sum() - C.diag().sum()) / (m * (m - 1))
    centered = M - M.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered)
    return float(off_diag_mean), float(eigenvalues[-1] / eigenvalues.sum()), float(M.norm(dim=1).mean())


def pair_statistics(E, vocab, pairs, eligible):
    cosines, ranks = [], []
    for a, b in pairs:
        cosines.append(float(E[vocab[a]] @ E[vocab[b]]))
        for x, y in ((a, b), (b, a)):
            sims = (E @ E[vocab[x]]).masked_fill(~eligible, -2.0)
            sims[vocab[x]] = -2.0
            ranks.append(int((sims > sims[vocab[y]]).sum().item()) + 1)
    return cosines, ranks


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab, freq = load_vocab()
    eligible = freq >= MIN_FREQ
    eligible[0] = False

    all_words = [w for p in RELATED_PAIRS + CONTROL_PAIRS for w in p] + RANDOM_WORDS
    for w in all_words:
        if w not in vocab or freq[vocab[w]] < MIN_FREQ:
            raise ValueError(f"Từ {w!r} thiếu trong vocab hoặc tần suất < {MIN_FREQ}")

    summary, pair_rows, neighbor_rows = [], [], []

    for name, ckpt_path, csv_path, n_params in MODELS:
        if not ckpt_path.exists():
            print(f"[BỎ QUA] {name}: chưa có {ckpt_path.name}")
            continue

        print(f"[{name}] {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "vocab" in ckpt and ckpt["vocab"] != vocab:
            raise ValueError(f"Vocabulary của {name} khác file vocab hiện tại.")

        M = find_input_embedding(ckpt["model_state_dict"], len(vocab))
        spaces = {
            "raw": F.normalize(M, dim=1),
            "centered": F.normalize(M - M.mean(dim=0, keepdim=True), dim=1),
        }

        losses, seconds = read_losses(csv_path)
        cos_mean, pc1, norm_mean = geometry(M)
        row = {
            "model": name,
            "epochs": ckpt.get("epoch"),
            "parameters": n_params,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "minutes_per_epoch": (sum(seconds) / len(seconds) / 60) if seconds else None,
            "geom_cosine_mean_2000": cos_mean,
            "geom_pc1_variance_ratio": pc1,
            "norm_mean": norm_mean,
        }

        for space, E in spaces.items():
            rel_cos, rel_ranks = pair_statistics(E, vocab, RELATED_PAIRS, eligible)
            ctl_cos, _ = pair_statistics(E, vocab, CONTROL_PAIRS, eligible)
            row[f"{space}_related_cos"] = sum(rel_cos) / len(rel_cos)
            row[f"{space}_control_cos"] = sum(ctl_cos) / len(ctl_cos)
            row[f"{space}_gap"] = row[f"{space}_related_cos"] - row[f"{space}_control_cos"]
            row[f"{space}_hit_at_10"] = sum(r <= 10 for r in rel_ranks) / len(rel_ranks)
            row[f"{space}_mrr"] = sum(1.0 / r for r in rel_ranks) / len(rel_ranks)

            for group, pairs, cosines in (("related", RELATED_PAIRS, rel_cos), ("control", CONTROL_PAIRS, ctl_cos)):
                for (a, b), c in zip(pairs, cosines):
                    pair_rows.append({"model": name, "space": space, "group": group,
                                      "word_a": a, "word_b": b, "cosine": c})

            for word in RANDOM_WORDS:
                sims = (E @ E[vocab[word]]).masked_fill(~eligible, -2.0)
                sims[vocab[word]] = -2.0
                top = torch.topk(sims, TOP_K)
                for rank, (score, idx) in enumerate(zip(top.values, top.indices), start=1):
                    token = next(t for t, i in vocab.items() if i == int(idx))
                    neighbor_rows.append({"model": name, "space": space, "word": word,
                                          "rank": rank, "neighbor": token, "cosine": float(score)})

        summary.append(row)

    if not summary:
        raise SystemExit("Không có checkpoint nào để đánh giá.")

    for filename, rows in (("summary.csv", summary), ("pairs.csv", pair_rows), ("neighbors.csv", neighbor_rows)):
        with (OUT_DIR / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metrics = [k for k in summary[0] if k != "model"]
    print("\n" + "=" * 100)
    print(f"{'chỉ số':30s}" + "".join(f"{r['model']:>17s}" for r in summary))
    for k in metrics:
        print(f"{k:30s}" + "".join(f"{fmt(r[k]):>17s}" for r in summary))

    print("\n" + "=" * 100)
    print(f"TOP-{TOP_K} LÁNG GIỀNG (không gian raw, chỉ xét từ có tần suất >= {MIN_FREQ})")
    for name in [r["model"] for r in summary]:
        print(f"\n### {name}")
        for word in RANDOM_WORDS:
            items = [r for r in neighbor_rows
                     if r["model"] == name and r["space"] == "raw" and r["word"] == word]
            print(f"  {word}: " + ", ".join(f"{r['neighbor']}({r['cosine']:.2f})" for r in items))

    print(f"\nĐã ghi: {OUT_DIR / 'summary.csv'}, pairs.csv, neighbors.csv")


if __name__ == "__main__":
    main()
