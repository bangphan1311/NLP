import json
import unicodedata
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/processed")
VOCAB_DIR = DATA_DIR / "vocab"
MIN_FREQ = 5


def normalize_token(token: str) -> str:
    token = token.lower()

    while token and unicodedata.category(token[0]).startswith("P") and token[0] != "_":
        token = token[1:]
    while token and unicodedata.category(token[-1]).startswith("P") and token[-1] != "_":
        token = token[:-1]

    if any(char.isdigit() for char in token):
        return ""

    return token


def build_vocab(corpus_path: Path, output_path: Path):
    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy corpus: {corpus_path}")

    counter = Counter()
    total_tokens = 0

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document["text"]

            for raw_token in text.split():
                total_tokens += 1
                token = normalize_token(raw_token)
                if token:
                    counter[token] += 1

    vocab = {"<unk>": 0}
    frequencies = {}

    for token, freq in counter.most_common():
        if freq >= MIN_FREQ and token not in vocab:
            vocab[token] = len(vocab)
            frequencies[token] = freq

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"vocab": vocab, "frequencies": frequencies},
            f,
            ensure_ascii=False,
            indent=2,
        )

    kept_tokens = sum(frequencies.values())
    print(f"Corpus: {corpus_path}")
    print(f"Raw tokens: {total_tokens:,}")
    print(f"Unique tokens after normalization: {len(counter):,}")
    print(f"Vocabulary size (including <unk>): {len(vocab):,}")
    print(f"Tokens retained by frequency: {kept_tokens:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    build_vocab(
        DATA_DIR / "corpus_unsegmented.jsonl",
        VOCAB_DIR / "unsegmented_vocab.json",
    )
    build_vocab(
        DATA_DIR / "corpus_segmented.jsonl",
        VOCAB_DIR / "segmented_vocab.json",
    )
