from typing import Dict, List, Tuple, Optional,ClassVar,Iterable,Iterator
import pickle
import multiprocessing as mp 
import sys
from cs336_basics.pretokenization_example import parallel_pretokenize, pretokenize_string
from collections import defaultdict
import json 



def build_pair_counts(word_storage, word_freqs):
    pair_counts = defaultdict(int)
    for word_idx, word in enumerate(word_storage):
        freq = word_freqs[word_idx]
        for pos in range(len(word) - 1):
            pair = (word[pos], word[pos+1])
            pair_counts[pair] += freq
    return pair_counts

def get_max_pair(pair_dict):
    if not pair_dict:
        return None
    max_count = max(pair_dict.values())
    max_pairs = [pair for pair, count in pair_dict.items() if count == max_count]
    if len(max_pairs) == 1:
        return max_pairs[0]
    # Break ties by selecting the lexicographically greater pair
    return max(max_pairs)

def merge(best_pair, word_storage, word_freqs, pair_counts):
    token1, token2 = best_pair
    new_token = token1 + token2

    for i in range(len(word_storage)):
        word = word_storage[i]
        freq = word_freqs[i]
        
        # Skip if this word doesn't contain the pair
        if len(word) < 2:
            continue
            
        merge_positions = []
        j = 0
        while j < len(word) - 1:
            if word[j] == token1 and word[j+1] == token2:
                merge_positions.append(j)
                j += 2  # Skip the next position since we'll merge these
            else:
                j += 1
        
        # If no merges needed for this word, continue
        if not merge_positions:
            continue
            
        # we do reverse because it keeps the indices -- neat! Ive tried the opposite and it was a pain in the ass 
        for j in reversed(merge_positions):
            # Decrement counts for pairs that will be broken
            if j > 0:
                left_pair = (word[j-1], token1)
                if left_pair in pair_counts:
                    pair_counts[left_pair] -= freq
                    if pair_counts[left_pair] <= 0:
                        del pair_counts[left_pair]
            
            if j < len(word) - 2:
                right_pair = (token2, word[j+2])
                if right_pair in pair_counts:
                    pair_counts[right_pair] -= freq
                    if pair_counts[right_pair] <= 0:
                        del pair_counts[right_pair]
            
            # Perform the merge
            word[j] = new_token
            word.pop(j+1)
            
            # Increment counts for new pairs created by the merge
            if j > 0:
                new_left_pair = (word[j-1], new_token)
                pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + freq
            
            if j < len(word) - 1:
                new_right_pair = (new_token, word[j+1])
                pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + freq
    
    return new_token

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given input text file.
    
    Args:
        input_path: Path to a text file with BPE tokenizer training data
        vocab_size: A positive integer that defines the maximum final vocabulary size 
                   (including the initial byte vocabulary, vocabulary items produced 
                   from merging, and any special tokens)
        special_tokens: A list of strings to add to the vocabulary. These special 
                       tokens do not otherwise affect BPE training
    
    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) 
               to bytes (token bytes)
        merges: A list of BPE merges produced from training. Each list item is a tuple 
                of bytes (<token1>, <token2>), representing that <token1> was merged 
                with <token2>. The merges should be ordered by order of creation
    """
    assert vocab_size >= 256
    pre_token_counts = parallel_pretokenize(input_path, num_processes=8)

    words_with_counts = []
    for word, count in pre_token_counts.items():
        word_bytes = word.encode('utf-8')
        tokens = [bytes([b]) for b in word_bytes]
        words_with_counts.append((tokens, count))

    vocab = {i: bytes([i]) for i in range(256)}
    
    # Add special tokens - they get IDs starting from 256
    for i, token_str in enumerate(special_tokens):
        token_bytes = token_str.encode('utf-8')
        vocab[256 + i] = token_bytes

    word_storage = [word for word, _ in words_with_counts]
    word_freqs = [count for _, count in words_with_counts]

    pair_counts = build_pair_counts(word_storage, word_freqs)
    
    merges = []
    
    num_merges = vocab_size - len(vocab)

    for i in range(num_merges):
        if not pair_counts:
            break
        
        best_pair = get_max_pair(pair_counts)
        if best_pair is None:
            break

        # Remove the best_pair from counts BEFORE merging
        if best_pair in pair_counts:
            del pair_counts[best_pair]

        # Pass pair_counts to merge
        new_token = merge(best_pair, word_storage, word_freqs, pair_counts)
        merges.append(best_pair)
        
        # Add new token to vocabulary with the next available ID
        vocab[len(vocab)] = new_token

    return vocab, merges

class Tokenizer:
    """
    A tokenizer that encodes text into integer IDs and decodes integer IDs back into text,
    using a vocabulary and a list of merges learned from BPE.
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """
        Constructs a Tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: A dictionary mapping token IDs (integers) to token bytes.
            merges: A list of byte-pair merges, where each merge is a tuple of two bytes.
            special_tokens: An optional list of special tokens (strings) to add to the vocabulary.
        """
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        self.reverse_lookup={}
        for id,token in self.vocab.items():
            self.reverse_lookup[token]=id
        
    
    @classmethod
    def from_files(cls,vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None) -> "Tokenizer":
        """
        Constructs a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath: Path to the serialized vocabulary file (e.g., JSON).(int -> byte)
            merges_filepath: Path to the serialized merges file (e.g., text file).
            special_tokens: An optional list of special tokens (strings) to add to the vocabulary.

        Returns:
            A Tokenizer instance.
        """
        with open(vocab_filepath,'r') as f: 
            vocab_data = json.load(f)
        vocab={}
        for key, value in vocab_data.items():
    
            token_id = int(key)
            vocab[token_id] = bytes(value)
        merges = []
        with open(merges_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Each line should contain two tokens separated by space
                    parts = line.split(' ', 1)  # Split on first space only
                    if len(parts) == 2:
                        token1 = parts[0].encode('utf-8')
                        token2 = parts[1].encode('utf-8')
                        merges.append((token1, token2))

        if special_tokens:
            for token_str in special_tokens:
                token_bytes = token_str.encode('utf-8')
                if token_bytes not in [v for v in vocab.values()]:
                    vocab[len(vocab)] = token_bytes
        
        return cls(vocab, merges, special_tokens)


    

    def encode(self, text: str) -> List[int]:
        """
        Encodes an input text into a sequence of token IDs.

        Args:
            text: The input text (string) to encode.

        Returns:
            A list of integer token IDs.
        """
        result_ids = []
        pretokens = pretokenize_string(text, self.special_tokens)
        
        # Process each chunk separately to avoid cross-chunk merges
        for chunk in pretokens:
            chunk_tokens = []
            if self.special_tokens is not None and chunk in self.special_tokens:
                chunk_tokens = [chunk.encode('utf-8')]
            else:
                encoded_string = chunk.encode('utf-8')
                chunk_tokens = [bytes([b]) for b in encoded_string]
            
            #  merges within this chunk only
            for merge in self.merges:
                token1, token2 = merge
                new_tokens = []
                i = 0
                while i < len(chunk_tokens):
                    if (i < len(chunk_tokens) - 1) and chunk_tokens[i] == token1 and chunk_tokens[i+1] == token2:
                        new_token = token1 + token2
                        i += 2
                    else:
                        new_token = chunk_tokens[i]
                        i += 1
                    new_tokens.append(new_token)
                chunk_tokens = new_tokens
            
            # Convert tokens to IDs for this chunk
            for token in chunk_tokens:
                if token in self.reverse_lookup:
                    result_ids.append(self.reverse_lookup[token])
    
        return result_ids
        
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization
        of large files that we cannot directly load into memory.

        Args:
            iterable: An iterable of strings (e.g., a file handle).

        Yields:
            Integer token IDs.
        """
        for text in iterable: 
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
        


    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs into text.

        Args:
            ids: A list of integer token IDs to decode.

        Returns:
            The decoded text (string)
        """
        replacement_bytes = b'\xef\xbf\xbd'
        all_bytes=b""
        for tok in ids:
            if tok not in self.vocab:
                all_bytes += (replacement_bytes)
            else: 
                all_bytes+= self.vocab[tok]
        #replacement bytes is the same when you do replace
        return all_bytes.decode('utf-8',errors="replace")
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    vocab, merges = train_bpe("/Users/ziyadhassan/lms/assignment1/tests/fixtures/tinystories_sample_5M.txt", 500, ['<|endoftext|>'])
    tok=Tokenizer(vocab,merges,['<|endoftext|>'])
    encoded= (tok.encode('hey this is me <|endoftext|> ziyad'))
    print(tok.decode(encoded))