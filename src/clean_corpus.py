import html
import json
import re
import unicodedata
from pathlib import Path

INPUT_PATH = Path("data/processed/corpus.jsonl")
OUTPUT_PATH = Path("data/processed/corpus_cleaned.jsonl")
MIN_TEXT_LENGTH = 200


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", html.unescape(text))

    # Remove comments, URLs, HTML tags and remaining wiki markup.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"<(ref|gallery|math|syntaxhighlight)\b[^>]*>.*?</\1\s*>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove common residual wiki syntax.
    text = re.sub(r"__[A-Z][A-Z0-9_]*__", " ", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?://\S+)\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"'{2,5}", "", text)

    # Remove leftover template delimiters and formatting artifacts.
    text = re.sub(r"\{\{|\}\}|\{\||\|\}", " ", text)
    text = re.sub(r"[{}\[\]]", " ", text)
    text = re.sub(r"(?m)^\s*=+\s*(.*?)\s*=+\s*$", r"\1", text)

    cleaned_lines = []
    previous_line = None

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Discard lines that contain no letters or digits.
        if not re.search(r"[A-Za-zÀ-ỹ0-9]", line):
            continue

        # Remove duplicate adjacent lines.
        if line == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = line

    return "\n".join(cleaned_lines).strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0
    removed = 0

    with INPUT_PATH.open("r", encoding="utf-8") as source, \
            OUTPUT_PATH.open("w", encoding="utf-8") as target:

        for line in source:
            if not line.strip():
                continue

            total += 1

            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue

            title = clean_text(document.get("title", ""))
            text = clean_text(document.get("text", ""))

            if len(text) < MIN_TEXT_LENGTH:
                removed += 1
                continue

            document["title"] = title
            document["text"] = text

            target.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )
            saved += 1

    print(f"Documents đầu vào: {total}")
    print(f"Documents giữ lại: {saved}")
    print(f"Documents bị loại: {removed}")
    print(f"File đầu ra: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()