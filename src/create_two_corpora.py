import json
from pathlib import Path

from underthesea import word_tokenize


INPUT_PATH = Path("data/processed/corpus_cleaned.jsonl")
OUTPUT_DIR = Path("data/processed")
OUTPUT_UNSEGMENTED = OUTPUT_DIR / "corpus_unsegmented.jsonl"
OUTPUT_SEGMENTED = OUTPUT_DIR / "corpus_segmented.jsonl"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    with (
        INPUT_PATH.open("r", encoding="utf-8") as source,
        OUTPUT_UNSEGMENTED.open("w", encoding="utf-8", newline="\n") as unsegmented_file,
        OUTPUT_SEGMENTED.open("w", encoding="utf-8", newline="\n") as segmented_file,
    ):
        for line in source:
            if not line.strip():
                continue

            document = json.loads(line)
            title = document.get("title", "")
            text = document.get("text", "")

            # Giữ nguyên văn bản để làm corpus không tách từ.
            unsegmented_record = {
                "title": title,
                "text": text,
            }
            unsegmented_file.write(
                json.dumps(unsegmented_record, ensure_ascii=False) + "\n"
            )

            # Tách từ tiếng Việt; dấu gạch dưới nối các thành phần của một từ ghép.
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
                print(f"Done {total:,} documents")

  
    print(f"Số documents: {total:,}")
    print(f"Corpus không tách từ: {OUTPUT_UNSEGMENTED}")
    print(f"Corpus có tách từ: {OUTPUT_SEGMENTED}")

    # Thử
    with OUTPUT_UNSEGMENTED.open("r", encoding="utf-8") as file:
        original = json.loads(next(file))
    with OUTPUT_SEGMENTED.open("r", encoding="utf-8") as file:
        segmented = json.loads(next(file))

    print("\n--- MẪU KHÔNG TÁCH TỪ ---")
    print(original["text"][:500])
    print("\n--- MẪU CÓ TÁCH TỪ ---")
    print(segmented["text"][:500])


if __name__ == "__main__":
    main()
