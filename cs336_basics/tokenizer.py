from typing import Dict, List, Tuple, Optional, Iterable, Iterator
import pickle

class Tokenizer:
    """
    A BPE tokenizer that encodes text into integer IDs and decodes integer IDs into text.
    Supports user-provided special tokens.
    """
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab: Dictionary mapping token IDs to their byte representations
            merges: List of byte pair merges to apply during encoding
            special_tokens: Optional list of special token strings to add to vocabulary
        """
        pass
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        """
        Class method that constructs and returns a Tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to the serialized vocabulary file
            merges_filepath: Path to the serialized merges file  
            special_tokens: Optional list of special token strings to add to vocabulary
            
        Returns:
            Tokenizer instance constructed from the files
        """
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: Input text string to encode
            
        Returns:
            List of token IDs representing the encoded text
        """
        pass
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This is required for memory-efficient tokenization 
        of large files that cannot directly be loaded into memory.
        
        Args:
            iterable: Iterable of strings to encode
            
        Yields:
            Token IDs one at a time
        """
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text
        """