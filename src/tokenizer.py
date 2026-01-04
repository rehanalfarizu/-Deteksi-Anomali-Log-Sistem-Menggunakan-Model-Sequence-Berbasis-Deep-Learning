"""
Tokenizer Module untuk Deteksi Anomali Log Sistem

Module ini berisi class Tokenizer untuk mengkonversi teks log
menjadi sequence numerik yang dapat diproses oleh model deep learning.

Author: Mahasiswa AI
Date: 2026-01-04
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LogTokenizer:
    """
    Tokenizer khusus untuk log sistem.
    
    Attributes:
        max_words (int): Jumlah maksimum kata dalam vocabulary
        max_length (int): Panjang maksimum sequence
        oov_token (str): Token untuk kata yang tidak dikenal
    """
    
    def __init__(
        self,
        max_words: int = 10000,
        max_length: int = 50,
        oov_token: str = "<OOV>",
        padding: str = "post",
        truncating: str = "post"
    ):
        """
        Inisialisasi tokenizer.
        
        Args:
            max_words: Jumlah maksimum kata dalam vocabulary
            max_length: Panjang maksimum sequence
            oov_token: Token untuk kata yang tidak dikenal
            padding: Posisi padding ('pre' atau 'post')
            truncating: Posisi truncating ('pre' atau 'post')
        """
        self.max_words = max_words
        self.max_length = max_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating
        
        self.tokenizer = KerasTokenizer(
            num_words=max_words,
            oov_token=oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '
        )
        
        self.word_index = None
        self.index_word = None
        self.vocabulary_size = None
        self.is_fitted = False
        
    def fit(self, texts: List[str]) -> 'LogTokenizer':
        """
        Fit tokenizer pada teks.
        
        Args:
            texts: List teks untuk training tokenizer
            
        Returns:
            self
        """
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        self.vocabulary_size = min(len(self.word_index) + 1, self.max_words)
        self.is_fitted = True
        
        return self
    
    def transform(
        self,
        texts: List[str],
        return_attention_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform teks menjadi sequences.
        
        Args:
            texts: List teks untuk ditransform
            return_attention_mask: Return attention mask
            
        Returns:
            Array sequences atau tuple (sequences, attention_mask)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer belum di-fit. Panggil fit() terlebih dahulu.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding=self.padding,
            truncating=self.truncating
        )
        
        if return_attention_mask:
            # Attention mask: 1 untuk token valid, 0 untuk padding
            attention_mask = (padded_sequences != 0).astype(np.int32)
            return padded_sequences, attention_mask
        
        return padded_sequences
    
    def fit_transform(
        self,
        texts: List[str],
        return_attention_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit tokenizer dan transform teks.
        
        Args:
            texts: List teks
            return_attention_mask: Return attention mask
            
        Returns:
            Array sequences atau tuple (sequences, attention_mask)
        """
        self.fit(texts)
        return self.transform(texts, return_attention_mask)
    
    def inverse_transform(self, sequences: np.ndarray) -> List[str]:
        """
        Convert sequences kembali ke teks.
        
        Args:
            sequences: Array sequences
            
        Returns:
            List teks
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer belum di-fit.")
        
        texts = []
        for seq in sequences:
            words = []
            for idx in seq:
                if idx > 0 and idx in self.index_word:
                    words.append(self.index_word[idx])
            texts.append(' '.join(words))
        
        return texts
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Dapatkan vocabulary.
        
        Returns:
            Dictionary word to index
        """
        return self.word_index
    
    def get_vocabulary_size(self) -> int:
        """
        Dapatkan ukuran vocabulary.
        
        Returns:
            Ukuran vocabulary
        """
        return self.vocabulary_size
    
    def save(self, filepath: str):
        """
        Simpan tokenizer ke file.
        
        Args:
            filepath: Path file untuk menyimpan
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Simpan konfigurasi dan state
        config = {
            "max_words": self.max_words,
            "max_length": self.max_length,
            "oov_token": self.oov_token,
            "padding": self.padding,
            "truncating": self.truncating,
            "vocabulary_size": self.vocabulary_size,
            "is_fitted": self.is_fitted
        }
        
        # Simpan config sebagai JSON
        config_path = filepath + ".config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Simpan tokenizer sebagai pickle
        tokenizer_path = filepath + ".tokenizer.pkl"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LogTokenizer':
        """
        Load tokenizer dari file.
        
        Args:
            filepath: Path file tokenizer
            
        Returns:
            Instance LogTokenizer
        """
        # Load config
        config_path = filepath + ".config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load tokenizer
        tokenizer_path = filepath + ".tokenizer.pkl"
        with open(tokenizer_path, "rb") as f:
            keras_tokenizer = pickle.load(f)
        
        # Buat instance baru
        instance = cls(
            max_words=config["max_words"],
            max_length=config["max_length"],
            oov_token=config["oov_token"],
            padding=config["padding"],
            truncating=config["truncating"]
        )
        
        instance.tokenizer = keras_tokenizer
        instance.word_index = keras_tokenizer.word_index
        instance.index_word = keras_tokenizer.index_word
        instance.vocabulary_size = config["vocabulary_size"]
        instance.is_fitted = config["is_fitted"]
        
        return instance


class CharacterTokenizer:
    """
    Tokenizer berbasis karakter untuk log.
    """
    
    def __init__(
        self,
        max_length: int = 200,
        lowercase: bool = True
    ):
        """
        Inisialisasi character tokenizer.
        
        Args:
            max_length: Panjang maksimum sequence
            lowercase: Konversi ke lowercase
        """
        self.max_length = max_length
        self.lowercase = lowercase
        
        # Karakter yang valid
        self.chars = (
            list('abcdefghijklmnopqrstuvwxyz') +
            list('0123456789') +
            list(' .,;:!?-_/\\@#$%&*()[]{}+=<>') +
            ['<PAD>', '<UNK>']
        )
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.vocabulary_size = len(self.chars)
        self.pad_idx = self.char_to_idx['<PAD>']
        self.unk_idx = self.char_to_idx['<UNK>']
        
    def encode(self, text: str) -> np.ndarray:
        """
        Encode teks ke sequence karakter.
        
        Args:
            text: Teks input
            
        Returns:
            Array indices karakter
        """
        if self.lowercase:
            text = text.lower()
        
        # Convert ke indices
        indices = []
        for char in text[:self.max_length]:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.unk_idx)
        
        # Padding
        if len(indices) < self.max_length:
            indices.extend([self.pad_idx] * (self.max_length - len(indices)))
        
        return np.array(indices)
    
    def decode(self, indices: np.ndarray) -> str:
        """
        Decode sequence kembali ke teks.
        
        Args:
            indices: Array indices
            
        Returns:
            Teks hasil decode
        """
        chars = []
        for idx in indices:
            if idx == self.pad_idx:
                break
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in ['<PAD>', '<UNK>']:
                    chars.append(char)
        
        return ''.join(chars)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch teks.
        
        Args:
            texts: List teks
            
        Returns:
            Array 2D indices
        """
        return np.array([self.encode(text) for text in texts])


class SequenceGenerator:
    """
    Generator untuk membuat sequence data untuk training.
    """
    
    def __init__(
        self,
        tokenizer: LogTokenizer,
        sequence_length: int = 50,
        step: int = 1
    ):
        """
        Inisialisasi sequence generator.
        
        Args:
            tokenizer: Instance LogTokenizer
            sequence_length: Panjang sequence
            step: Step size untuk sliding window
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.step = step
        
    def create_sequences(
        self,
        logs: List[str],
        labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Buat sequences dari daftar log.
        
        Args:
            logs: List log messages
            labels: List labels (opsional)
            
        Returns:
            Tuple (X, y) sequences dan labels
        """
        # Gabungkan semua log menjadi satu teks
        all_text = ' [SEP] '.join(logs)
        
        # Tokenize
        tokens = self.tokenizer.tokenizer.texts_to_sequences([all_text])[0]
        
        # Buat sequences dengan sliding window
        X = []
        y = []
        
        for i in range(0, len(tokens) - self.sequence_length, self.step):
            X.append(tokens[i:i + self.sequence_length])
            # Untuk language model, y adalah token berikutnya
            if i + self.sequence_length < len(tokens):
                y.append(tokens[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def create_classification_sequences(
        self,
        logs: List[str],
        labels: List[int],
        window_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Buat sequences untuk klasifikasi anomali.
        
        Args:
            logs: List log messages
            labels: List labels (0=normal, 1=anomali)
            window_size: Ukuran window log
            
        Returns:
            Tuple (X, y)
        """
        X = []
        y = []
        
        for i in range(0, len(logs) - window_size + 1):
            window_logs = logs[i:i + window_size]
            window_labels = labels[i:i + window_size]
            
            # Gabungkan log dalam window
            combined_log = ' [SEP] '.join(window_logs)
            
            # Tokenize
            sequence = self.tokenizer.tokenizer.texts_to_sequences([combined_log])[0]
            
            # Pad/truncate
            if len(sequence) < self.tokenizer.max_length:
                sequence = sequence + [0] * (self.tokenizer.max_length - len(sequence))
            else:
                sequence = sequence[:self.tokenizer.max_length]
            
            X.append(sequence)
            
            # Label: 1 jika ada anomali dalam window
            y.append(1 if max(window_labels) == 1 else 0)
        
        return np.array(X), np.array(y)


if __name__ == "__main__":
    # Contoh penggunaan
    sample_logs = [
        "Failed password for invalid user admin from 192.168.1.100 port 22",
        "Accepted password for user from 192.168.1.50 port 22",
        "systemd started nginx service",
        "Connection received from 192.168.1.75",
        "ERROR Database connection failed",
    ]
    
    # Test LogTokenizer
    print("=" * 50)
    print("LogTokenizer Test")
    print("=" * 50)
    
    tokenizer = LogTokenizer(max_words=1000, max_length=20)
    sequences = tokenizer.fit_transform(sample_logs)
    
    print(f"Vocabulary size: {tokenizer.get_vocabulary_size()}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"\nSample sequence:\n{sequences[0]}")
    
    # Inverse transform
    reconstructed = tokenizer.inverse_transform(sequences)
    print(f"\nReconstructed: {reconstructed[0]}")
    
    # Test CharacterTokenizer
    print("\n" + "=" * 50)
    print("CharacterTokenizer Test")
    print("=" * 50)
    
    char_tokenizer = CharacterTokenizer(max_length=50)
    encoded = char_tokenizer.batch_encode(sample_logs)
    
    print(f"Vocabulary size: {char_tokenizer.vocabulary_size}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"\nSample encoded: {encoded[0][:20]}")
    
    # Decode
    decoded = char_tokenizer.decode(encoded[0])
    print(f"Decoded: {decoded}")
    
    # Test SequenceGenerator
    print("\n" + "=" * 50)
    print("SequenceGenerator Test")
    print("=" * 50)
    
    labels = [1, 0, 0, 0, 1]  # 1 = anomali, 0 = normal
    
    generator = SequenceGenerator(tokenizer, sequence_length=15)
    X, y = generator.create_classification_sequences(sample_logs, labels, window_size=3)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Labels: {y}")
