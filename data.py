import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import IterableDataset


def content_to_text(x):
    return {"text": x["content"]}


def stack_language(language, access_token):
    return load_dataset(
        "bigcode/starcoderdata",
        data_dir=language,
        streaming=True,
        split="train",
        token=access_token,
    ).select_columns(["content"]).map(content_to_text, remove_columns=["content"])


def create_dataset(access_token):
    python_data = stack_language("python", access_token)
    typescript_data = stack_language("typescript", access_token)
    rust_data = stack_language("rust", access_token)
    sql_data = stack_language("sql", access_token)
    shell_data = stack_language("shell", access_token)

    fineweb_edu = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        streaming=True,
        split="train",
        token=access_token,
    ).select_columns(["text"])

    finemath = load_dataset(
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        streaming=True,
        split="train",
        token=access_token,
    ).select_columns(["text"])

    return interleave_datasets(
        [python_data, finemath, fineweb_edu, typescript_data, rust_data, sql_data, shell_data],
        probabilities=[0.50, 0.18, 0.12, 0.08, 0.05, 0.04, 0.03],
        seed=42,
        stopping_strategy="first_exhausted",
    )


class CoderDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, context_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.eos_token_id = tokenizer.eos_token_id

        if self.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id.")

    def __iter__(self):
        buffer = []

        for example in self.dataset:
            encoded = self.tokenizer.encode(example["text"])
            encoded.append(self.eos_token_id)
            buffer.extend(encoded)

            while len(buffer) >= self.context_length + 1:
                x = torch.tensor(buffer[0 : self.context_length], dtype=torch.long)
                y = torch.tensor(buffer[1 : self.context_length + 1], dtype=torch.long)
                buffer = buffer[self.context_length:]
                yield x, y
