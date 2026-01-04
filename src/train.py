"""
Training Script untuk Deteksi Anomali Log Sistem

Script ini digunakan untuk melatih model deep learning
untuk mendeteksi anomali pada log sistem.

Author: Mahasiswa AI
Date: 2026-01-04
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import SystemLogGenerator
from src.preprocessing import LogPreprocessor, preprocess_dataframe
from src.tokenizer import LogTokenizer
from src.model import create_model, get_callbacks


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load konfigurasi dari file YAML.
    
    Args:
        config_path: Path ke file konfigurasi
        
    Returns:
        Dictionary konfigurasi
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(
    config: Dict[str, Any],
    generate_new: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LogTokenizer]:
    """
    Persiapkan data untuk training.
    
    Args:
        config: Dictionary konfigurasi
        generate_new: Generate data baru atau load dari file
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test, tokenizer)
    """
    print("\n" + "="*50)
    print("PERSIAPAN DATA")
    print("="*50)
    
    # Generate atau load data
    if generate_new:
        print("\nGenerating synthetic data...")
        generator = SystemLogGenerator(
            seed=config['generator']['random_seed'],
            anomaly_ratio=config['generator']['anomaly_ratio']
        )
        df = generator.generate_logs(config['generator']['num_logs'])
    else:
        # Load dari file
        train_path = os.path.join(config['data']['processed_dir'], config['data']['train_file'])
        df = pd.read_csv(train_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Normal samples: {len(df[df['label'] == 0])}")
    print(f"Anomaly samples: {len(df[df['label'] == 1])}")
    
    # Preprocessing
    print("\nPreprocessing logs...")
    preprocessor = LogPreprocessor(
        lowercase=config['preprocessing']['lowercase'],
        remove_special_chars=config['preprocessing']['remove_special_chars'],
        remove_numbers=config['preprocessing']['remove_numbers']
    )
    df = preprocess_dataframe(df, log_column='log_message', preprocessor=preprocessor)
    
    # Tokenization
    print("\nTokenizing logs...")
    tokenizer = LogTokenizer(
        max_words=config['data']['vocab_size'],
        max_length=config['data']['max_sequence_length']
    )
    
    X = tokenizer.fit_transform(df['processed_log'].tolist())
    y = df['label'].values
    
    print(f"Vocabulary size: {tokenizer.get_vocabulary_size()}")
    print(f"Sequence shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config['generator']['random_seed'],
        stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, tokenizer


def build_model(config: Dict[str, Any], vocab_size: int) -> Any:
    """
    Build model berdasarkan konfigurasi.
    
    Args:
        config: Dictionary konfigurasi
        vocab_size: Ukuran vocabulary
        
    Returns:
        Model instance
    """
    print("\n" + "="*50)
    print("BUILD MODEL")
    print("="*50)
    
    model_type = config['model']['type']
    print(f"\nModel type: {model_type.upper()}")
    
    # Buat model
    if model_type == 'bilstm':
        model = create_model(
            model_type='bilstm',
            vocab_size=vocab_size,
            embedding_dim=config['model']['embedding_dim'],
            max_length=config['data']['max_sequence_length'],
            lstm_units=config['model']['lstm_units'],
            dense_units=config['model']['dense_units'],
            dropout_rate=config['model']['dropout_rate'],
            recurrent_dropout=config['model']['recurrent_dropout'],
            use_attention=True
        )
    else:
        model = create_model(
            model_type=model_type,
            vocab_size=vocab_size,
            embedding_dim=config['model']['embedding_dim'],
            max_length=config['data']['max_sequence_length'],
            lstm_units=config['model']['lstm_units'] if model_type == 'lstm' else None,
            gru_units=config['model']['lstm_units'] if model_type == 'gru' else None,
            dense_units=config['model']['dense_units'],
            dropout_rate=config['model']['dropout_rate'],
            recurrent_dropout=config['model']['recurrent_dropout']
        )
    
    # Build dan compile
    model.build()
    model.compile(
        learning_rate=config['training']['learning_rate'],
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )
    
    model.summary()
    
    return model


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> Any:
    """
    Training model.
    
    Args:
        model: Model instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Konfigurasi
        
    Returns:
        Training history
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Setup callbacks
    callbacks = get_callbacks(
        model_dir=config['output']['model_dir'],
        log_dir=config['output']['log_dir'],
        patience=config['training']['early_stopping']['patience'],
        monitor=config['training']['early_stopping']['monitor']
    )
    
    # Training
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Evaluasi model.
    
    Args:
        model: Model instance
        X_test, y_test: Test data
        config: Konfigurasi
        save_plots: Simpan visualisasi
        
    Returns:
        Dictionary hasil evaluasi
    """
    print("\n" + "="*50)
    print("EVALUASI MODEL")
    print("="*50)
    
    # Prediksi
    y_pred_proba = model.model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    results = {}
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    results['roc_auc'] = roc_auc
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    results['pr_auc'] = pr_auc
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    results['accuracy'] = accuracy
    results['precision'] = precision_score
    results['recall'] = recall_score
    results['f1_score'] = f1
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall: {recall_score:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if save_plots:
        viz_dir = config['output']['visualization_dir']
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Plot ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'roc_curve.png'), dpi=150)
        plt.close()
        
        # Plot Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'pr_curve.png'), dpi=150)
        plt.close()
        
        print(f"\nVisualization saved to {viz_dir}")
    
    return results


def plot_training_history(history, config: Dict[str, Any]):
    """
    Plot training history.
    
    Args:
        history: Training history
        config: Konfigurasi
    """
    viz_dir = config['output']['visualization_dir']
    os.makedirs(viz_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"Training history plot saved to {viz_dir}")


def save_results(results: Dict[str, Any], tokenizer: LogTokenizer, config: Dict[str, Any]):
    """
    Simpan hasil training.
    
    Args:
        results: Dictionary hasil evaluasi
        tokenizer: Tokenizer yang sudah di-fit
        config: Konfigurasi
    """
    # Simpan results
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Simpan tokenizer
    model_dir = config['output']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    tokenizer.save(tokenizer_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new synthetic data')
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set random seed
    tf.random.set_seed(config['generator']['random_seed'])
    np.random.seed(config['generator']['random_seed'])
    
    # Prepare data
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(
        config, generate_new=args.generate_data or True
    )
    
    # Split training untuk validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config['training']['validation_split'],
        random_state=config['generator']['random_seed'],
        stratify=y_train
    )
    
    # Build model
    model = build_model(config, tokenizer.get_vocabulary_size())
    
    # Training
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    
    # Plot training history
    plot_training_history(history, config)
    
    # Evaluasi
    results = evaluate_model(model, X_test, y_test, config)
    
    # Simpan hasil
    save_results(results, tokenizer, config)
    
    # Simpan model final
    model_path = os.path.join(config['output']['model_dir'], 'final_model.keras')
    model.save(model_path)
    
    print("\n" + "="*50)
    print("TRAINING SELESAI!")
    print("="*50)
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {config['output']['results_dir']}")


if __name__ == "__main__":
    main()
