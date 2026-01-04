"""
Model Deep Learning untuk Deteksi Anomali Log Sistem

Module ini berisi definisi arsitektur model LSTM, GRU, dan Bi-LSTM
untuk mendeteksi anomali pada log sistem.

Author: Mahasiswa AI
Date: 2026-01-04
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, GRU, Bidirectional,
    Dense, Dropout, BatchNormalization, Attention,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate,
    Conv1D, MaxPooling1D, Flatten, LayerNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam


class AnomalyDetectionModel:
    """
    Base class untuk model deteksi anomali.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        max_length: int = 50,
        dropout_rate: float = 0.3
    ):
        """
        Inisialisasi model.
        
        Args:
            vocab_size: Ukuran vocabulary
            embedding_dim: Dimensi embedding
            max_length: Panjang maksimum sequence
            dropout_rate: Dropout rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build(self):
        """Build model. Harus diimplementasikan di subclass."""
        raise NotImplementedError
        
    def compile(
        self,
        learning_rate: float = 0.001,
        loss: str = "binary_crossentropy",
        metrics: List[str] = None
    ):
        """
        Compile model.
        
        Args:
            learning_rate: Learning rate
            loss: Loss function
            metrics: List metrik evaluasi
        """
        if metrics is None:
            metrics = ['accuracy']
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model belum di-build.")
            
    def save(self, filepath: str):
        """Simpan model."""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Model belum di-build.")
            
    def load(self, filepath: str):
        """Load model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class LSTMModel(AnomalyDetectionModel):
    """
    Model LSTM untuk deteksi anomali.
    
    Arsitektur:
    - Embedding Layer
    - LSTM Layers (stacked)
    - Dropout
    - Dense Layers
    - Output Layer (sigmoid)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        max_length: int = 50,
        lstm_units: List[int] = None,
        dense_units: int = 16,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        l2_reg: float = 0.01
    ):
        """
        Inisialisasi LSTM model.
        
        Args:
            vocab_size: Ukuran vocabulary
            embedding_dim: Dimensi embedding
            max_length: Panjang maksimum sequence
            lstm_units: List unit untuk setiap LSTM layer
            dense_units: Unit untuk dense layer
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            l2_reg: L2 regularization strength
        """
        super().__init__(vocab_size, embedding_dim, max_length, dropout_rate)
        
        self.lstm_units = lstm_units or [64, 32]
        self.dense_units = dense_units
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        
    def build(self) -> Model:
        """
        Build LSTM model.
        
        Returns:
            Keras Model
        """
        # Input
        inputs = Input(shape=(self.max_length,), name='input')
        
        # Embedding
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'lstm_{i+1}'
            )(x)
            
            if return_sequences:
                x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Dense layers
        x = Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense'
        )(x)
        x = Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_AnomalyDetector')
        
        return self.model


class GRUModel(AnomalyDetectionModel):
    """
    Model GRU untuk deteksi anomali.
    
    GRU memiliki arsitektur yang lebih sederhana dari LSTM
    namun sering memberikan performa yang comparable.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        max_length: int = 50,
        gru_units: List[int] = None,
        dense_units: int = 16,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        l2_reg: float = 0.01
    ):
        """
        Inisialisasi GRU model.
        """
        super().__init__(vocab_size, embedding_dim, max_length, dropout_rate)
        
        self.gru_units = gru_units or [64, 32]
        self.dense_units = dense_units
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        
    def build(self) -> Model:
        """
        Build GRU model.
        
        Returns:
            Keras Model
        """
        # Input
        inputs = Input(shape=(self.max_length,), name='input')
        
        # Embedding
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            x = GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'gru_{i+1}'
            )(x)
            
            if return_sequences:
                x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Dense layers
        x = Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense'
        )(x)
        x = Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='GRU_AnomalyDetector')
        
        return self.model


class BiLSTMModel(AnomalyDetectionModel):
    """
    Model Bidirectional LSTM untuk deteksi anomali.
    
    Bi-LSTM memproses sequence dari kedua arah (forward dan backward)
    untuk menangkap konteks lebih baik.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        max_length: int = 50,
        lstm_units: List[int] = None,
        dense_units: int = 16,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        l2_reg: float = 0.01,
        use_attention: bool = False
    ):
        """
        Inisialisasi Bi-LSTM model.
        
        Args:
            use_attention: Gunakan attention mechanism
        """
        super().__init__(vocab_size, embedding_dim, max_length, dropout_rate)
        
        self.lstm_units = lstm_units or [64, 32]
        self.dense_units = dense_units
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        
    def build(self) -> Model:
        """
        Build Bi-LSTM model.
        
        Returns:
            Keras Model
        """
        # Input
        inputs = Input(shape=(self.max_length,), name='input')
        
        # Embedding
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1 or self.use_attention
            x = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=regularizers.l2(self.l2_reg)
                ),
                name=f'bilstm_{i+1}'
            )(x)
            
            if return_sequences and not (i == len(self.lstm_units) - 1 and self.use_attention):
                x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Attention (opsional)
        if self.use_attention:
            # Self-attention
            attention = Attention(name='attention')([x, x])
            x = GlobalAveragePooling1D(name='global_pool')(attention)
        
        # Dense layers
        x = Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense'
        )(x)
        x = Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_AnomalyDetector')
        
        return self.model


class CNNLSTMModel(AnomalyDetectionModel):
    """
    Model CNN-LSTM hybrid untuk deteksi anomali.
    
    CNN digunakan untuk ekstraksi fitur lokal,
    LSTM untuk menangkap dependensi temporal.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        max_length: int = 50,
        cnn_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        lstm_units: int = 64,
        dense_units: int = 16,
        dropout_rate: float = 0.3
    ):
        """
        Inisialisasi CNN-LSTM model.
        """
        super().__init__(vocab_size, embedding_dim, max_length, dropout_rate)
        
        self.cnn_filters = cnn_filters or [64, 128]
        self.kernel_sizes = kernel_sizes or [3, 3]
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        
    def build(self) -> Model:
        """
        Build CNN-LSTM model.
        
        Returns:
            Keras Model
        """
        # Input
        inputs = Input(shape=(self.max_length,), name='input')
        
        # Embedding
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # CNN layers
        for i, (filters, kernel_size) in enumerate(zip(self.cnn_filters, self.kernel_sizes)):
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            x = MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_cnn_{i+1}')(x)
        
        # LSTM
        x = LSTM(
            units=self.lstm_units,
            dropout=self.dropout_rate,
            name='lstm'
        )(x)
        
        # Dense layers
        x = Dense(self.dense_units, activation='relu', name='dense')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='CNNLSTM_AnomalyDetector')
        
        return self.model


class AutoencoderModel(AnomalyDetectionModel):
    """
    Model Autoencoder LSTM untuk deteksi anomali.
    
    Autoencoder ditraining untuk merekonstruksi input normal.
    Anomali terdeteksi ketika reconstruction error tinggi.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        max_length: int = 50,
        encoder_units: List[int] = None,
        latent_dim: int = 32,
        dropout_rate: float = 0.3
    ):
        """
        Inisialisasi Autoencoder model.
        """
        super().__init__(vocab_size, embedding_dim, max_length, dropout_rate)
        
        self.encoder_units = encoder_units or [64, 32]
        self.latent_dim = latent_dim
        self.threshold = None
        
    def build(self) -> Model:
        """
        Build Autoencoder model.
        
        Returns:
            Keras Model
        """
        # Input
        inputs = Input(shape=(self.max_length,), name='input')
        
        # Embedding
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # Encoder
        for i, units in enumerate(self.encoder_units):
            return_sequences = i < len(self.encoder_units) - 1
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                name=f'encoder_lstm_{i+1}'
            )(x)
        
        # Latent space
        encoded = Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        # Decoder
        x = Dense(self.encoder_units[-1], activation='relu', name='decoder_dense')(encoded)
        x = tf.expand_dims(x, axis=1)
        x = tf.repeat(x, repeats=self.max_length, axis=1)
        
        for i, units in enumerate(reversed(self.encoder_units)):
            return_sequences = True
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                name=f'decoder_lstm_{i+1}'
            )(x)
        
        # Output reconstruction
        outputs = Dense(self.embedding_dim, activation='linear', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder')
        
        return self.model
    
    def set_threshold(self, X_normal: np.ndarray, percentile: float = 95):
        """
        Set threshold berdasarkan data normal.
        
        Args:
            X_normal: Data normal untuk menghitung threshold
            percentile: Percentile untuk threshold
        """
        # Embed input untuk comparison
        embedding_layer = self.model.get_layer('embedding')
        embedded = embedding_layer(X_normal)
        
        # Get reconstruction
        reconstructed = self.model.predict(X_normal)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(embedded.numpy() - reconstructed), axis=(1, 2))
        
        self.threshold = np.percentile(mse, percentile)
        print(f"Threshold set to: {self.threshold}")
        
    def detect_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Deteksi anomali berdasarkan reconstruction error.
        
        Args:
            X: Input data
            
        Returns:
            Array prediksi (1=anomali, 0=normal)
        """
        if self.threshold is None:
            raise ValueError("Threshold belum di-set. Panggil set_threshold() terlebih dahulu.")
        
        embedding_layer = self.model.get_layer('embedding')
        embedded = embedding_layer(X)
        reconstructed = self.model.predict(X)
        
        mse = np.mean(np.square(embedded.numpy() - reconstructed), axis=(1, 2))
        
        return (mse > self.threshold).astype(int)


def create_model(
    model_type: str,
    vocab_size: int,
    **kwargs
) -> AnomalyDetectionModel:
    """
    Factory function untuk membuat model.
    
    Args:
        model_type: Jenis model ('lstm', 'gru', 'bilstm', 'cnn_lstm', 'autoencoder')
        vocab_size: Ukuran vocabulary
        **kwargs: Parameter tambahan untuk model
        
    Returns:
        Instance model
    """
    models = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'bilstm': BiLSTMModel,
        'cnn_lstm': CNNLSTMModel,
        'autoencoder': AutoencoderModel
    }
    
    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' tidak dikenali. Pilih dari: {list(models.keys())}")
    
    return models[model_type](vocab_size=vocab_size, **kwargs)


def get_callbacks(
    model_dir: str = "models/saved_models",
    log_dir: str = "logs",
    patience: int = 5,
    monitor: str = "val_loss"
) -> List:
    """
    Dapatkan callbacks untuk training.
    
    Args:
        model_dir: Direktori untuk menyimpan model
        log_dir: Direktori untuk logs
        patience: Patience untuk early stopping
        monitor: Metrik yang dimonitor
        
    Returns:
        List callbacks
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Contoh penggunaan
    print("=" * 50)
    print("Model Architecture Demo")
    print("=" * 50)
    
    # Parameter
    vocab_size = 10000
    embedding_dim = 128
    max_length = 50
    
    # Build dan tampilkan setiap model
    models_to_test = ['lstm', 'gru', 'bilstm', 'cnn_lstm']
    
    for model_type in models_to_test:
        print(f"\n{model_type.upper()} Model:")
        print("-" * 30)
        
        model = create_model(
            model_type=model_type,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length
        )
        model.build()
        model.compile()
        model.summary()
