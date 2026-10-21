import json
import unicodedata
from pathlib import Path

from underthesea import word_tokenize

INPUT_PATH = Path("data/processed/corpus_cleaned.jsonl")
OUTPUT_DIR = Path("data/processed")
OUTPUT_UNSEGMENTED = OUTPUT_DIR / "corpus_unsegmented.jsonl"
OUTPUT_SEGMENTED = OUTPUT_DIR / "corpus_segmented.jsonl"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_PATH}. "
            "Hãy chạy src/clean_corpus.py trước."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    with (
        INPUT_PATH.open("r", encoding="utf-8") as source,
        OUTPUT_UNSEGMENTED.open(
            "w", encoding="utf-8", newline="\n"
        ) as unsegmented_file,
        OUTPUT_SEGMENTED.open(
            "w", encoding="utf-8", newline="\n"
        ) as segmented_file,
    ):
        for line in source:
            if not line.strip():
                continue

            document = json.loads(line)
            title = document.get("title", "")
            text = unicodedata.normalize(
                "NFC", document.get("text", "")
            )

            if not text.strip():
                continue

            unsegmented_record = {
                "title": title,
                "text": text,
            }
            unsegmented_file.write(
                json.dumps(unsegmented_record, ensure_ascii=False) + "\n"
            )

            segmented_text = word_tokenize(text, format="text")
            segmented_record = {
                "title": title,
                "text": segmented_text,
            }
            segmented_file.write(
                json.dumps(segmented_record, ensure_ascii=False) + "\n"
            )

            total += 1
            if total % 1000 == 0:
                print(f"Đã xử lý {total:,} documents")

    print(f"Tổng documents: {total:,}")
    print(f"Corpus không tách từ: {OUTPUT_UNSEGMENTED}")
    print(f"Corpus có tách từ: {OUTPUT_SEGMENTED}")


if __name__ == "__main__":
    main()
