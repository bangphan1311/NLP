import json
from pathlib import Path

DATA_DIR = Path("data/processed")
VOCAB_DIR = DATA_DIR / "vocab"
OUTPUT_DIR = DATA_DIR / "cbow"

CORPORA = {
    "unsegmented": (
        DATA_DIR / "corpus_unsegmented.jsonl",
        VOCAB_DIR / "unsegmented_vocab.json",
    ),
    "segmented": (
        DATA_DIR / "corpus_segmented.jsonl",
        VOCAB_DIR / "segmented_vocab.json",
    ),
}


def encode_corpus(corpus_path: Path, vocab_path: Path, output_path: Path):
    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy corpus: {corpus_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]

    unk_id = vocab["<unk>"]
    total_docs = 0
    total_tokens = 0
    unk_tokens = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        corpus_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue

            document = json.loads(line)
            ids = []

            for raw_token in document["text"].split():
                token = raw_token.lower()

                import unicodedata

                while (
                    token
                    and unicodedata.category(token[0]).startswith("P")
                    and token[0] != "_"
                ):
                    token = token[1:]

                while (
                    token
                    and unicodedata.category(token[-1]).startswith("P")
                    and token[-1] != "_"
                ):
                    token = token[:-1]

                if not token or any(ch.isdigit() for ch in token):
                    continue

                token_id = vocab.get(token, unk_id)
                ids.append(token_id)

                if token_id == unk_id:
                    unk_tokens += 1

            if ids:
                target.write(json.dumps(ids) + "\n")
                total_docs += 1
                total_tokens += len(ids)

    print(f"Corpus: {corpus_path.name}")
    print(f"Documents encoded: {total_docs:,}")
    print(f"Tokens encoded: {total_tokens:,}")
    print(f"Tokens mapped to <unk>: {unk_tokens:,}")
    print(f"Output: {output_path}")


def main():
    for name, (corpus_path, vocab_path) in CORPORA.items():
        encode_corpus(
            corpus_path,
            vocab_path,
            OUTPUT_DIR / f"{name}_ids.jsonl",
        )


if __name__ == "__main__":
    main()
