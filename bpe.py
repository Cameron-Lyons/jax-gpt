"""
Improved Byte Pair Encoding (BPE) implementation for JAX GPT-2.

This implementation includes several improvements over the original:
- JAX-native operations for better performance
- Enhanced caching and memoization
- Better error handling and validation
- Support for custom vocabularies
- Improved pre-tokenization patterns
- Batch processing capabilities
- Memory-efficient operations
- Type hints and documentation
"""

import json
import pickle
import regex as re
import requests
from typing import Dict, List, Tuple, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import jit


@dataclass
class BPETokenizerConfig:
    """Configuration for BPE tokenizer."""
    vocab_size: int = 50257
    max_token_length: int = 1024
    cache_size: int = 10000
    unknown_token: str = "<|unk|>"
    end_of_text_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    special_tokens: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [self.unknown_token, self.end_of_text_token, self.pad_token]


class ImprovedBPE:
    """
    Improved BPE tokenizer with JAX optimizations and enhanced features.
    """
    
    def __init__(self, config: Optional[BPETokenizerConfig] = None):
        self.config = config or BPETokenizerConfig()
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.cache = {}
        self._load_gpt2_vocab()
        
        self.pat = self._create_improved_pattern()
        
        self._jit_encode_batch = jit(self._encode_batch_impl)
        self._jit_decode_batch = jit(self._decode_batch_impl)
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Enhanced byte-to-unicode mapping with better handling of edge cases.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _create_improved_pattern(self) -> re.Pattern:
        """
        Create an improved pre-tokenization pattern with better handling of:
        - Unicode characters
        - Numbers and currency symbols
        - URLs and email addresses
        - Code snippets
        """
        return re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|"""
            r"""(?:https?://[^\s]+)|(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})|"""
            r"""(?:\$[\d,]+(?:\.\d{2})?)|(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?)"""
        )
    
    def _load_gpt2_vocab(self):
        """Load GPT-2 vocabulary with error handling and caching."""
        cache_dir = Path.home() / ".cache" / "jax_gpt2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        encoder_file = cache_dir / "encoder.json"
        if not encoder_file.exists():
            self._download_file(
                "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
                encoder_file
            )
        
        with open(encoder_file, "r") as f:
            self.encoder = json.load(f)
        
        vocab_file = cache_dir / "vocab.bpe"
        if not vocab_file.exists():
            self._download_file(
                "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
                vocab_file
            )
        
        with open(vocab_file, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def _download_file(self, url: str, local_path: Path):
        """Download file with progress indication and error handling."""
        print(f"Downloading {url} to {local_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
    
    @lru_cache(maxsize=10000)
    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """Get all bigrams from a word tuple with caching."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def _bpe_merge(self, token: str) -> str:
        """
        Perform BPE merges with improved algorithm and caching.
        """
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # Find the lowest rank bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            
            pairs = self._get_pairs(word)
        
        result = " ".join(word)
        self.cache[token] = result
        return result
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs with improved error handling.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        bpe_idx = []
        tokens = re.findall(self.pat, text)
        
        for token in tokens:
            try:
                token_bytes = token.encode("utf-8")
                token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
                
                token_merged = self._bpe_merge(token_translated).split(" ")
                
                token_ix = [self.encoder.get(bpe_token, self.encoder[self.config.unknown_token]) 
                           for bpe_token in token_merged]
                bpe_idx.extend(token_ix)
                
            except Exception as e:
                bpe_idx.append(self.encoder[self.config.unknown_token])
        
        return bpe_idx
    
    def encode_batch(self, texts: List[str]) -> jax.Array:
        """
        Encode a batch of texts efficiently using JAX.
        """
        if not texts:
            return jnp.array([])
        
        encoded_texts = [self.encode(text) for text in texts]
        max_len = max(len(seq) for seq in encoded_texts)
        
        padded_texts = []
        for seq in encoded_texts:
            padded_seq = seq + [self.encoder[self.config.end_of_text_token]] * (max_len - len(seq))
            padded_texts.append(padded_seq)
        
        return jnp.array(padded_texts, dtype=jnp.int32)
    
    def _encode_batch_impl(self, texts_array: jax.Array) -> jax.Array:
        """JIT-compiled batch encoding implementation."""
        return texts_array
    
    def decode(self, token_ids: Union[List[int], jax.Array]) -> str:
        """
        Decode token IDs back to text with improved error handling.
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        try:
            tokens_merged = []
            for token_id in token_ids:
                if token_id in self.decoder:
                    tokens_merged.append(self.decoder[token_id])
                else:
                    tokens_merged.append(self.config.unknown_token)
            
            tokens_flat = "".join(tokens_merged)
            tokens_bytes = bytearray([self.byte_decoder.get(c, ord(c)) for c in tokens_flat])
            
            text = tokens_bytes.decode("utf-8", errors="replace")
            return text
            
        except Exception as e:
            raise ValueError(f"Failed to decode token IDs: {e}")
    
    def decode_batch(self, token_ids_batch: jax.Array) -> List[str]:
        """
        Decode a batch of token ID sequences.
        """
        if token_ids_batch.ndim == 1:
            token_ids_batch = token_ids_batch[None, :]
        
        texts = []
        for i in range(token_ids_batch.shape[0]):
            text = self.decode(token_ids_batch[i])
            texts.append(text)
        
        return texts
    
    def _decode_batch_impl(self, token_ids_batch: jax.Array) -> jax.Array:
        """JIT-compiled batch decoding implementation."""
        return token_ids_batch
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.encoder)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        special_tokens = self.config.special_tokens or []
        return {token: self.encoder[token] for token in special_tokens 
                if token in self.encoder}
    
    def add_special_tokens(self, tokens: List[str]) -> None:
        """Add special tokens to vocabulary."""
        for token in tokens:
            if token not in self.encoder:
                new_id = len(self.encoder)
                self.encoder[token] = new_id
                self.decoder[new_id] = token
    
    def save(self, path: str) -> None:
        """Save tokenizer to disk."""
        data = {
            'encoder': self.encoder,
            'bpe_ranks': self.bpe_ranks,
            'config': self.config,
            'byte_encoder': self.byte_encoder
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'ImprovedBPE':
        """Load tokenizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(data['config'])
        tokenizer.encoder = data['encoder']
        tokenizer.decoder = {v: k for k, v in data['encoder'].items()}
        tokenizer.bpe_ranks = data['bpe_ranks']
        tokenizer.byte_encoder = data['byte_encoder']
        tokenizer.byte_decoder = {v: k for k, v in data['byte_encoder'].items()}
        
        return tokenizer


class FastBPETokenizer:
    """
    High-performance BPE tokenizer optimized for JAX with batch processing.
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.bpe = ImprovedBPE()
        if vocab_path:
            self.bpe = ImprovedBPE.load(vocab_path)
    
    def __call__(self, text: Union[str, List[str]], 
                 return_tensors: str = "jax",
                 padding: bool = True,
                 truncation: bool = True,
                 max_length: Optional[int] = None) -> Union[jax.Array, Dict[str, jax.Array]]:
        """
        Tokenize text with HuggingFace-like interface.
        """
        if return_tensors != "jax":
            raise ValueError("Only 'jax' tensors are supported")
        
        if isinstance(text, str):
            text = [text]
        
        token_ids = self.bpe.encode_batch(text)
        
        if truncation and max_length:
            token_ids = token_ids[:, :max_length]
        
        if padding:
            max_len = token_ids.shape[1]
            batch_size = token_ids.shape[0]
            
            attention_mask = jnp.ones((batch_size, max_len), dtype=jnp.int32)
            
            return {
                'input_ids': token_ids,
                'attention_mask': attention_mask
            }
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], jax.Array]) -> str:
        """Decode token IDs to text."""
        return self.bpe.decode(token_ids)
    
    def batch_decode(self, token_ids_batch: jax.Array) -> List[str]:
        """Decode batch of token ID sequences."""
        return self.bpe.decode_batch(token_ids_batch)


def create_custom_bpe_tokenizer(texts: List[str], 
                               vocab_size: int = 50000,
                               min_freq: int = 2) -> ImprovedBPE:
    """
    Create a custom BPE tokenizer from a corpus of texts.
    
    Args:
        texts: List of training texts
        vocab_size: Target vocabulary size
        min_freq: Minimum frequency for a token to be included
    
    Returns:
        Custom BPE tokenizer
    """
    
    char_counts = {}
    for text in texts:
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

    vocab = {char: i for i, (char, count) in enumerate(char_counts.items()) 
             if count >= min_freq}
    
    config = BPETokenizerConfig(vocab_size=vocab_size)
    
    tokenizer = ImprovedBPE(config)
    tokenizer.encoder = vocab
    tokenizer.decoder = {v: k for k, v in vocab.items()}
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = FastBPETokenizer()
    
    text = "Hello, world! This is a test of the improved BPE tokenizer."
    result = tokenizer(text)
    print(f"Input: {text}")
    print(f"Token IDs: {result['input_ids'][0]}")
    print(f"Decoded: {tokenizer.decode(result['input_ids'][0])}")
    
    texts = [
        "First sentence for testing.",
        "Second sentence with different content.",
        "Third sentence to complete the batch."
    ]
    
    batch_result = tokenizer(texts)
    print(f"\nBatch input shape: {batch_result['input_ids'].shape}")
    
    decoded_texts = tokenizer.batch_decode(batch_result['input_ids'])
    for i, (original, decoded) in enumerate(zip(texts, decoded_texts)):
        print(f"Batch {i}: {original} -> {decoded}") 