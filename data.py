import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def create_python_dataset(data_file):
    return load_dataset(
        "json",
        data_files=data_file,
        split="train",
    )["text"]


class CoderDataset(Dataset):
    def __init__(self, dataset, tokenizer, context_length):
        all_tokens = []
        
        if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define eos_token_id.")

        for start in range(0, len(dataset), 100):
            texts = dataset[start : start + 100]
            encoded_batch = tokenizer(texts)["input_ids"]

            for encoded in encoded_batch:
                all_tokens.extend(encoded)
                all_tokens.append(tokenizer.eos_token_id)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long) 
        self.context_length = context_length

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, index):
        start = index * self.context_length
        chunk = self.tokens[start : start + self.context_length + 1]
        return chunk[:-1], chunk[1:]