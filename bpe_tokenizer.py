from collections import defaultdict
from collections.abc import Mapping

Pair = tuple[int, int]

def pair_freq_counter(ids: list[int]) -> defaultdict[tuple[int, int], int]:
    freq_counter = defaultdict(int)
    for idx in range(len(ids) - 1):
        curr = ids[idx]
        next_token = ids[idx + 1]
        freq_counter[curr, next_token] += 1
    return freq_counter


def select_pair(
        freq_counter: Mapping[Pair, int],
        min_frequency: int = 2
) -> Pair | None:
    
    if not freq_counter:
        return None 
    
    most_freq_pair = max(freq_counter.keys(), key=lambda x:freq_counter[x])
    if freq_counter[most_freq_pair] < min_frequency:
        return None
    
    return most_freq_pair

def apply_merge(
        ids: list[int],
        pair_to_merge: tuple[int, int],
        new_id: int
) -> list[int]:
    
    new_ids = []
    idx = 0
    # Manually loop over all adjacent pairs and apply merge if applicable
    while idx + 1 < len(ids):
        curr = ids[idx]
        next_token = ids[idx + 1]
        
        if curr == pair_to_merge[0] and next_token == pair_to_merge[1]:
            new_ids.append(new_id)
            idx += 2
        else:
            new_ids.append(curr)
            idx += 1
    # Add the last token to the new sequence
    if idx < len(ids):
        new_ids.append(ids[idx])
    
    return new_ids


class BPETokenizer():
    def __init__(self):
        self.merges: dict[Pair, int] = {}
        # We want each token id to be converted into byte or bytes
        # The first 0-255 token ids already are directly their byte versions
        self.vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    
    def train(
            self,
            training_text: str,
            vocab_size: int,
            min_pair_frequency: int = 2,
    ) -> None:
        
        encoded = list(training_text.encode("utf-8"))
        encoded = self._apply_merge_rules(encoded)
        
        applied_merges = len(self.merges)
        curr_vocab_size = applied_merges + 256
        
        while curr_vocab_size < vocab_size:
            freq_counter = pair_freq_counter(encoded)
            most_freq_pair = select_pair(freq_counter, min_pair_frequency)
            if most_freq_pair is None:
                break

            new_token_id = applied_merges + 256
            
            self.merges[most_freq_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]
            
            encoded = apply_merge(encoded, most_freq_pair, new_token_id)
            applied_merges += 1
            curr_vocab_size += 1

    def encode(
            self,
            text: str,
    ) -> list[int]:
        ids = list(text.encode("utf-8"))
        return self._apply_merge_rules(ids)

    def decode(
            self,
            ids: list[int]
    ) -> str:
        
        byte_sequence = bytearray()
        for token_id in ids:
            byte_sequence.extend(self.vocab[token_id])
        
        return byte_sequence.decode("utf-8")


    def _apply_merge_rules(self, ids: list[int]) -> list[int]:
        for pair, new_id in self.merges.items():
            ids = apply_merge(ids, pair, new_id)
        return ids