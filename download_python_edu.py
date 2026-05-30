import argparse
import gzip
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


rows_url = "https://datasets-server.huggingface.co/rows"
contents_url = "https://softwareheritage.s3.amazonaws.com/content"


def get_examples(num_files):
    for offset in range(0, num_files, 100):
        query = urlencode(
            {
                "dataset": "HuggingFaceTB/smollm-corpus",
                "config": "python-edu",
                "split": "train",
                "offset": offset,
                "length": min(100, num_files - offset),
            }
        )
        with urlopen(f"{rows_url}?{query}") as response:
            page = json.load(response)
        for example in page["rows"]:
            yield example["row"]


def download_contents(blob_id):
    with urlopen(f"{contents_url}/{blob_id}") as response:
        return gzip.decompress(response.read()).decode("utf-8", errors="ignore")


def save_example(file, example):
    example["text"] = download_contents(example["blob_id"])
    file.write(json.dumps(example) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/python-edu")
    parser.add_argument("--train-files", type=int, default=5000)
    parser.add_argument("--val-files", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        (output_dir / "train.jsonl").open("w") as train_file,
        (output_dir / "val.jsonl").open("w") as val_file,
    ):
        for index, example in enumerate(get_examples(args.val_files + args.train_files)):
            if index < args.val_files:
                save_example(val_file, example)
            else:
                save_example(train_file, example)


if __name__ == "__main__":
    main()
