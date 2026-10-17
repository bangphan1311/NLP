
import html
import json
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "undertheseanlp/UVW-2026"
OUTPUT_PATH = Path("data/processed/corpus.jsonl")

NUM_DOCUMENTS = 10_000
MIN_DOCUMENT_LENGTH = 200


def clean_wikipedia_text(text: str) -> str:
    """Clean common Wikipedia markup and normalize article text."""

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Remove comments, references, and selected HTML blocks.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    text = re.sub(
        r"<(ref|gallery|math|syntaxhighlight)\b[^>]*>.*?</\1\s*>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<ref\b[^>]*/\s*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove nested Wikipedia templates iteratively.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)

    # Remove remaining template and table markup.
    text = re.sub(r"\{\{|\}\}", " ", text)
    text = re.sub(r"\{\|[\s\S]*?\|\}", " ", text)

    # Remove Wikipedia magic words, e.g. __NOTOC__.
    text = re.sub(
        r"__[A-Z][A-Z0-9_]*__",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove file, image, and category links.
    text = re.sub(
        r"\[\[(?:File|Image|Category|Tập tin|Hình|Thể loại):[^\]]*\]\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # [[target|display text]] -> display text.
    text = re.sub(
        r"\[\[([^\]|]+)\|([^\]]+)\]\]",
        r"\2",
        text,
    )

    # [[target]] -> target.
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # [https://example.com description] -> description.
    text = re.sub(
        r"\[(?:https?://\S+)\s+([^\]]+)\]",
        r"\1",
        text,
    )

    # Remove remaining URLs and HTML tags.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove wiki heading and bold/italic markers.
    text = re.sub(
        r"(?m)^\s*=+\s*(.*?)\s*=+\s*$",
        r"\1",
        text,
    )
    text = re.sub(r"'{2,5}", "", text)

    # Remove common table syntax and template-parameter lines.
    text = re.sub(
        r"(?m)^\s*(?:\|-|!\s*|[|}]\s?).*$",
        " ",
        text,
    )
    text = re.sub(
        r"(?m)^\s*\|\s*[^=\n]+\s*=.*$",
        " ",
        text,
    )

    # Remove leftover wiki brackets and braces.
    text = re.sub(r"[{}\[\]]", " ", text)

    # Normalize lines and discard empty or symbol-only lines.
    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        if not re.search(r"[A-Za-zÀ-ỹ]", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    scanned = 0
    saved = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_main_page = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        for article in dataset:
            if saved >= NUM_DOCUMENTS:
                break

            scanned += 1

            title = article.get("title", "")
            raw_content = article.get("content", "")

            if not isinstance(title, str):
                title = ""

            cleaned_content = clean_wikipedia_text(raw_content)

            # Skip the Wikipedia main page.
            if title.strip().casefold() in {
                "trang chính",
                "main page",
            }:
                skipped_main_page += 1
                continue

            # Skip empty or excessively short documents.
            if len(cleaned_content) < MIN_DOCUMENT_LENGTH:
                if cleaned_content:
                    skipped_short += 1
                else:
                    skipped_empty += 1
                continue

            # Add the article title to the document text.
            cleaned_title = clean_wikipedia_text(title)

            if cleaned_title:
                cleaned_content = f"{cleaned_title}\n{cleaned_content}"

            record = {
                "title": title,
                "text": cleaned_content,
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            saved += 1

            if saved % 1000 == 0:
                print(f"Saved {saved:,}/{NUM_DOCUMENTS:,} documents")

    print("\n========== CLEANING SUMMARY ==========")
    print(f"Articles scanned: {scanned:,}")
    print(f"Documents saved: {saved:,}")
    print(f"Main pages skipped: {skipped_main_page:,}")
    print(f"Empty documents skipped: {skipped_empty:,}")
    print(f"Short documents skipped: {skipped_short:,}")
    print(f"Output: {OUTPUT_PATH.resolve()}")

    if saved < NUM_DOCUMENTS:
        print(
            f"WARNING: Only {saved:,} valid documents were collected; "
            f"target was {NUM_DOCUMENTS:,}."
        )

    # Display a sample for manual inspection.
    if saved > 0:
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            sample = json.loads(next(file))

        print("\n========== CLEANED SAMPLE ==========")
        print("Title:", sample["title"])
        print(sample["text"][:1000])


if __name__ == "__main__":
    main()
