import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


def create_python_dataset(data_file):
    return load_dataset(
        "json",
        data_files=data_file,
        split="train",
        streaming=True,
    ).select_columns(["text"])


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
