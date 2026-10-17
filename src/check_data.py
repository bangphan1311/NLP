

from itertools import islice

from datasets import load_dataset


DATASET_NAME = "undertheseanlp/UVW-2026"


def main():
    print("Loading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    for index, article in enumerate(islice(dataset, 3), start=1):
        print(f"\n===== ARTICLE {index} =====")
        print("Available fields:", list(article.keys()))
        print("TITLE:", repr(article.get("title", "")))

        content = article.get("content", "")
        print("CONTENT:", repr(content[:500]))

    print("\nSource data check completed.")


if __name__ == "__main__":
    main()
