import csv
import json

import torch
import torch.nn.functional as F

from eval_words import CONTROL_PAIRS, MIN_FREQ, RELATED_PAIRS
from evaluate_compare import ROOT, find_input_embedding, read_losses

OUT_DIR = ROOT / "results/compare"
SEG = {
    "vocab": ROOT / "data/processed/vocab/segmented_vocab.json",
    "ckpt": ROOT / "models/compare/segmented_ns.pt",
    "csv": ROOT / "results/compare/segmented_ns.csv",
}
UNS = {
    "vocab": ROOT / "data/processed/vocab/unsegmented_vocab.json",
    "ckpt": ROOT / "models/compare/unsegmented_ns.pt",
    "csv": ROOT / "results/compare/unsegmented_ns.csv",
}
SHOW_WORDS = ["học_sinh", "bóng_đá", "máy_tính", "bác_sĩ", "hà_nội"]
TOP_K = 5


def load_vocab_file(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = data["vocab"]
    freq = torch.zeros(len(vocab))
    for token, token_id in vocab.items():
        if token_id != 0:
            freq[token_id] = float(data["frequencies"][token])
    return vocab, freq


def load_embeddings(spec, vocab):
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    if "vocab" in ckpt and ckpt["vocab"] != vocab:
        raise ValueError(f"Vocabulary không khớp với {spec['ckpt'].name}")
    return find_input_embedding(ckpt["model_state_dict"], len(vocab))


def evaluate(E, index, rel_pairs, ctl_pairs, center):
    X = E - E.mean(dim=0, keepdim=True) if center else E
    X = F.normalize(X, dim=1)

    def cosine(a, b):
        return float(X[index[a]] @ X[index[b]])

    ranks = []
    for a, b in rel_pairs:
        for x, y in ((a, b), (b, a)):
            sims = X @ X[index[x]]
            sims[index[x]] = -2.0
            ranks.append(int((sims > sims[index[y]]).sum().item()) + 1)

    rel = sum(cosine(a, b) for a, b in rel_pairs) / len(rel_pairs)
    ctl = sum(cosine(a, b) for a, b in ctl_pairs) / len(ctl_pairs)
    return {
        "related_cos": rel,
        "control_cos": ctl,
        "gap": rel - ctl,
        "hit_at_10": sum(r <= 10 for r in ranks) / len(ranks),
        "mrr": sum(1.0 / r for r in ranks) / len(ranks),
    }


def main():
    seg_vocab, seg_freq = load_vocab_file(SEG["vocab"])
    uns_vocab, uns_freq = load_vocab_file(UNS["vocab"])
    M_seg = load_embeddings(SEG, seg_vocab)
    M_uns = load_embeddings(UNS, uns_vocab)

    words = []
    for token, token_id in seg_vocab.items():
        if token_id == 0 or seg_freq[token_id] < MIN_FREQ:
            continue
        if all(s in uns_vocab for s in token.split("_")):
            words.append(token)

    index = {w: i for i, w in enumerate(words)}
    E_seg = torch.stack([M_seg[seg_vocab[w]] for w in words])
    E_uns = torch.stack(
        [M_uns[[uns_vocab[s] for s in w.split("_")]].mean(dim=0) for w in words]
    )

    needed = {w for p in RELATED_PAIRS + CONTROL_PAIRS for w in p}
    missing = needed - set(index)
    if missing:
        raise ValueError(f"Từ thiếu trong tập ứng viên chung: {missing}")

    def syllables(w):
        return set(w.split("_"))

    independent = [p for p in RELATED_PAIRS if not (syllables(p[0]) & syllables(p[1]))]
    print(f"Tập ứng viên chung: {len(words):,} từ")
    print(f"Cặp liên quan dùng chung âm tiết (bị loại ở dòng 'độc lập'): "
          f"{[p for p in RELATED_PAIRS if p not in independent]}\n")

    rows = []
    for space, E in (("tach_tu", E_seg), ("khong_tach_tu", E_uns)):
        for center in (False, True):
            for label, rel in (("10_cap", RELATED_PAIRS), ("cap_doc_lap", independent)):
                m = evaluate(E, index, rel, CONTROL_PAIRS, center)
                rows.append({"space": space, "centered": center, "pairs": label, **m})

    print(f"{'không gian':16s}{'centered':>9s}{'cặp':>13s}{'rel_cos':>9s}{'ctl_cos':>9s}{'gap':>8s}{'hit@10':>8s}{'mrr':>8s}")
    for r in rows:
        print(f"{r['space']:16s}{str(r['centered']):>9s}{r['pairs']:>13s}"
              f"{r['related_cos']:>9.4f}{r['control_cos']:>9.4f}{r['gap']:>8.4f}"
              f"{r['hit_at_10']:>8.2f}{r['mrr']:>8.4f}")

    print("\nTHỐNG KÊ HUẤN LUYỆN")
    stats = []
    for name, spec, vocab, freq in (("tach_tu", SEG, seg_vocab, seg_freq),
                                    ("khong_tach_tu", UNS, uns_vocab, uns_freq)):
        losses, seconds = read_losses(spec["csv"])
        item = {
            "space": name,
            "vocab_size": len(vocab),
            "tokens_kept": int(freq.sum()),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "minutes_per_epoch": sum(seconds) / len(seconds) / 60 if seconds else None,
        }
        stats.append(item)
        print(item)
    print("(Loss của hai corpus KHÔNG so trực tiếp: tập token đích khác nhau.)")

    print(f"\nTOP-{TOP_K} LÁNG GIỀNG (trong tập ứng viên chung, đã trừ vector trung bình)")
    for space, E in (("tach_tu", E_seg), ("khong_tach_tu", E_uns)):
        X = F.normalize(E - E.mean(dim=0, keepdim=True), dim=1)
        print(f"\n### {space}")
        for w in SHOW_WORDS:
            sims = X @ X[index[w]]
            sims[index[w]] = -2.0
            top = torch.topk(sims, TOP_K)
            print(f"  {w}: " + ", ".join(f"{words[int(i)]}({float(s):.2f})" for s, i in zip(top.values, top.indices)))

    for filename, data in (("tokenization_summary.csv", rows), ("tokenization_stats.csv", stats)):
        with (OUT_DIR / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    print(f"\nĐã ghi: {OUT_DIR / 'tokenization_summary.csv'}, tokenization_stats.csv")


if __name__ == "__main__":
    main()
