"""
Transformer-based Models untuk Deteksi Anomali Log Sistem

Module ini berisi implementasi arsitektur Transformer untuk
sequence classification pada log sistem.

Models:
1. TransformerEncoder - Custom Transformer encoder
2. TransformerClassifier - Full Transformer untuk klasifikasi
3. BERTLikeModel - BERT-style architecture untuk logs

Reference:
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017

Author: Mahasiswa AI
Date: 2026-01-06
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add
)
from typing import List, Tuple, Optional, Dict, Any


class PositionalEncoding(layers.Layer):
    """
    Positional Encoding layer untuk Transformer.
    
    Menambahkan informasi posisi ke embedding karena
    Transformer tidak memiliki informasi urutan secara inheren.
    """
    
    def __init__(self, max_length: int, embed_dim: int, **kwargs):
        """
        Args:
            max_length: Panjang maksimum sequence
            embed_dim: Dimensi embedding
        """
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Compute positional encodings
        self.pos_encoding = self._get_positional_encoding(max_length, embed_dim)
        
    def _get_positional_encoding(self, max_length: int, embed_dim: int) -> tf.Tensor:
        """Generate positional encoding matrix."""
        positions = np.arange(max_length)[:, np.newaxis]
        dimensions = np.arange(embed_dim)[np.newaxis, :]
        
        # Compute angles
        angles = positions / np.power(10000, (2 * (dimensions // 2)) / embed_dim)
        
        # Apply sin to even indices, cos to odd
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
            'embed_dim': self.embed_dim
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Single Transformer Encoder Block.
    
    Terdiri dari:
    1. Multi-Head Self-Attention
    2. Feed-Forward Network
    3. Layer Normalization
    4. Residual Connections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Args:
            embed_dim: Dimensi embedding
            num_heads: Jumlah attention heads
            ff_dim: Dimensi feed-forward layer
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-Forward Network
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dropout(dropout_rate),
            Dense(embed_dim),
            Dropout(dropout_rate)
        ])
        
        # Layer Normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, inputs, training=None, mask=None):
        """Forward pass."""
        # Self-Attention with residual connection
        attn_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-Forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    """
    Combined Token and Position Embedding layer.
    
    Menggabungkan word embedding dengan positional encoding.
    """
    
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        embed_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_emb = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )
        self.pos_emb = Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )
        
    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config


class TransformerClassifier(Model):
    """
    Transformer-based classifier untuk deteksi anomali log.
    
    Arsitektur:
    - Token + Positional Embedding
    - N x Transformer Encoder Blocks
    - Global Average Pooling
    - Classification Head
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_transformer_blocks: int = 2,
        dense_units: int = 64,
        dropout_rate: float = 0.1,
        num_classes: int = 1,
        **kwargs
    ):
        """
        Args:
            vocab_size: Ukuran vocabulary
            max_length: Panjang maksimum sequence
            embed_dim: Dimensi embedding
            num_heads: Jumlah attention heads
            ff_dim: Dimensi feed-forward
            num_transformer_blocks: Jumlah transformer blocks
            dense_units: Units di classification head
            dropout_rate: Dropout rate
            num_classes: Jumlah kelas output (1 untuk binary)
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = TokenAndPositionEmbedding(
            max_length=max_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_transformer_blocks)
        ]
        
        # Pooling
        self.pooling = GlobalAveragePooling1D()
        
        # Classification head
        self.dense1 = Dense(dense_units, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(
            num_classes,
            activation='sigmoid' if num_classes == 1 else 'softmax'
        )
        
    def call(self, inputs, training=None):
        """Forward pass."""
        # Embedding
        x = self.embedding(inputs)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Pooling
        x = self.pooling(x)
        
        # Classification
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        
        return outputs
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_blocks': self.num_transformer_blocks,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerWithCLS(Model):
    """
    Transformer dengan [CLS] token untuk klasifikasi (BERT-style).
    
    Menggunakan special [CLS] token di awal sequence
    untuk representasi seluruh sequence.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_transformer_blocks: int = 2,
        dense_units: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Special CLS token embedding
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer='random_normal',
            trainable=True
        )
        
        # Embeddings
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_length + 1, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]
        
        # Classification head
        self.dense = Dense(dense_units, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Token embeddings
        x = self.token_embedding(inputs)
        
        # Prepend CLS token
        cls_tokens = tf.broadcast_to(
            self.cls_token,
            [batch_size, 1, self.embed_dim]
        )
        x = tf.concat([cls_tokens, x], axis=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Use CLS token output for classification
        cls_output = x[:, 0, :]
        
        # Classification
        x = self.dense(cls_output)
        x = self.dropout(x, training=training)
        output = self.classifier(x)
        
        return output


class HybridCNNTransformer(Model):
    """
    Hybrid model menggabungkan CNN dan Transformer.
    
    CNN digunakan untuk ekstraksi fitur lokal,
    Transformer untuk dependensi jarak jauh.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 50,
        embed_dim: int = 128,
        cnn_filters: int = 64,
        kernel_sizes: List[int] = [3, 4, 5],
        num_heads: int = 4,
        ff_dim: int = 256,
        num_transformer_blocks: int = 1,
        dense_units: int = 64,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Embedding
        self.embedding = Embedding(vocab_size, embed_dim, input_length=max_length)
        
        # CNN layers (parallel convolutions with different kernel sizes)
        self.conv_layers = [
            layers.Conv1D(cnn_filters, kernel_size, activation='relu', padding='same')
            for kernel_size in kernel_sizes
        ]
        
        # Transformer block
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=cnn_filters * len(kernel_sizes),
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_transformer_blocks)
        ]
        
        # Output layers
        self.global_pool = GlobalAveragePooling1D()
        self.dense = Dense(dense_units, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=None):
        # Embedding
        x = self.embedding(inputs)
        
        # Parallel CNN
        conv_outputs = [conv(x) for conv in self.conv_layers]
        x = tf.concat(conv_outputs, axis=-1)
        
        # Transformer
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Classification
        x = self.global_pool(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        output = self.classifier(x)
        
        return output


def create_transformer_model(
    model_type: str,
    vocab_size: int,
    max_length: int = 50,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 2,
    dropout_rate: float = 0.1,
    **kwargs
) -> Model:
    """
    Factory function untuk membuat Transformer model.
    
    Args:
        model_type: Tipe model ('transformer', 'transformer_cls', 'cnn_transformer')
        vocab_size: Ukuran vocabulary
        max_length: Panjang maksimum sequence
        embed_dim: Dimensi embedding
        num_heads: Jumlah attention heads
        ff_dim: Dimensi feed-forward
        num_blocks: Jumlah transformer blocks
        dropout_rate: Dropout rate
        
    Returns:
        Keras Model
    """
    model_type = model_type.lower()
    
    if model_type == 'transformer':
        model = TransformerClassifier(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_blocks,
            dropout_rate=dropout_rate,
            **kwargs
        )
    elif model_type == 'transformer_cls':
        model = TransformerWithCLS(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_blocks,
            dropout_rate=dropout_rate,
            **kwargs
        )
    elif model_type == 'cnn_transformer':
        model = HybridCNNTransformer(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_blocks,
            dropout_rate=dropout_rate,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def build_transformer_model(
    vocab_size: int,
    max_length: int = 50,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 2,
    dense_units: int = 64,
    dropout_rate: float = 0.1
) -> Model:
    """
    Build Transformer model menggunakan Functional API.
    
    Alternative implementation using Keras Functional API.
    """
    # Input
    inputs = Input(shape=(max_length,), name='input')
    
    # Embedding
    x = TokenAndPositionEmbedding(
        max_length=max_length,
        vocab_size=vocab_size,
        embed_dim=embed_dim
    )(inputs)
    
    # Transformer blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f'transformer_block_{i}'
        )(x)
    
    # Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TransformerAnomalyDetector')
    
    return model


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Transformer Model for Log Anomaly Detection - Demo")
    print("="*60)
    
    # Parameters
    vocab_size = 10000
    max_length = 50
    embed_dim = 128
    
    # Create model
    print("\n1. Creating Transformer model...")
    model = create_transformer_model(
        model_type='transformer',
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=256,
        num_blocks=2,
        dropout_rate=0.1
    )
    
    # Build model dengan dummy input
    dummy_input = tf.random.uniform((1, max_length), maxval=vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    # Summary
    print("\nModel Summary:")
    model.summary()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Test prediction
    print("\n2. Testing prediction...")
    test_input = tf.random.uniform((4, max_length), maxval=vocab_size, dtype=tf.int32)
    predictions = model.predict(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.flatten()}")
    
    # Create other variants
    print("\n3. Creating model variants...")
    
    print("  - Transformer with CLS token...")
    model_cls = create_transformer_model('transformer_cls', vocab_size, max_length)
    _ = model_cls(dummy_input)
    print(f"    Parameters: {model_cls.count_params():,}")
    
    print("  - CNN-Transformer Hybrid...")
    model_hybrid = create_transformer_model('cnn_transformer', vocab_size, max_length)
    _ = model_hybrid(dummy_input)
    print(f"    Parameters: {model_hybrid.count_params():,}")
    
    print("\n✅ All models created successfully!")
