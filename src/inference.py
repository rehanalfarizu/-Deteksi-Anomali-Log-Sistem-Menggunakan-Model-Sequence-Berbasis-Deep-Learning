"""
Inference Script untuk Deteksi Anomali Log Sistem

Script ini digunakan untuk melakukan prediksi anomali
pada data log baru menggunakan model yang sudah ditraining.

Author: Mahasiswa AI
Date: 2026-01-04
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LogPreprocessor
from src.tokenizer import LogTokenizer


class AnomalyDetector:
    """
    Class untuk mendeteksi anomali pada log sistem.
    
    Attributes:
        model: Trained Keras model
        tokenizer: LogTokenizer yang sudah di-fit
        preprocessor: LogPreprocessor
        threshold: Threshold untuk klasifikasi
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        threshold: float = 0.5
    ):
        """
        Inisialisasi detector.
        
        Args:
            model_path: Path ke saved model
            tokenizer_path: Path ke saved tokenizer
            threshold: Threshold untuk klasifikasi (default: 0.5)
        """
        self.threshold = threshold
        self.preprocessor = LogPreprocessor(
            lowercase=True,
            remove_special_chars=True,
            remove_numbers=False
        )
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = LogTokenizer.load(tokenizer_path)
        print("Tokenizer loaded successfully!")
        
    def preprocess(self, logs: List[str]) -> np.ndarray:
        """
        Preprocess logs untuk prediksi.
        
        Args:
            logs: List log messages
            
        Returns:
            Array sequences yang siap untuk prediksi
        """
        # Preprocessing
        processed = self.preprocessor.preprocess(logs)
        
        # Tokenization
        sequences = self.tokenizer.transform(processed)
        
        return sequences
    
    def predict(self, logs: List[str]) -> Dict[str, Any]:
        """
        Prediksi anomali pada logs.
        
        Args:
            logs: List log messages
            
        Returns:
            Dictionary hasil prediksi
        """
        # Preprocess
        sequences = self.preprocess(logs)
        
        # Prediksi
        probabilities = self.model.predict(sequences, verbose=0)
        predictions = (probabilities > self.threshold).astype(int).flatten()
        
        # Hasil
        results = {
            "total_logs": len(logs),
            "anomaly_count": int(np.sum(predictions)),
            "normal_count": int(len(predictions) - np.sum(predictions)),
            "predictions": []
        }
        
        for i, (log, prob, pred) in enumerate(zip(logs, probabilities.flatten(), predictions)):
            results["predictions"].append({
                "index": i,
                "log": log[:100] + "..." if len(log) > 100 else log,
                "probability": float(prob),
                "prediction": "ANOMALY" if pred == 1 else "NORMAL",
                "is_anomaly": bool(pred == 1)
            })
        
        return results
    
    def predict_single(self, log: str) -> Dict[str, Any]:
        """
        Prediksi anomali untuk satu log.
        
        Args:
            log: Log message
            
        Returns:
            Dictionary hasil prediksi
        """
        result = self.predict([log])
        return result["predictions"][0]
    
    def predict_file(self, filepath: str, output_path: str = None) -> Dict[str, Any]:
        """
        Prediksi anomali dari file log.
        
        Args:
            filepath: Path ke file log
            output_path: Path untuk menyimpan hasil (opsional)
            
        Returns:
            Dictionary hasil prediksi
        """
        # Baca file
        with open(filepath, 'r') as f:
            logs = f.readlines()
        
        # Bersihkan newlines
        logs = [log.strip() for log in logs if log.strip()]
        
        print(f"Processing {len(logs)} logs from {filepath}...")
        
        # Prediksi
        results = self.predict(logs)
        
        # Simpan hasil jika output_path diberikan
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> str:
        """
        Analisis hasil prediksi.
        
        Args:
            results: Dictionary hasil prediksi
            
        Returns:
            String ringkasan analisis
        """
        total = results["total_logs"]
        anomalies = results["anomaly_count"]
        normal = results["normal_count"]
        
        analysis = []
        analysis.append("=" * 50)
        analysis.append("ANALISIS HASIL DETEKSI ANOMALI")
        analysis.append("=" * 50)
        analysis.append(f"\nTotal Log Dianalisis: {total}")
        analysis.append(f"Log Normal: {normal} ({normal/total*100:.1f}%)")
        analysis.append(f"Log Anomali: {anomalies} ({anomalies/total*100:.1f}%)")
        
        if anomalies > 0:
            analysis.append("\n" + "-" * 50)
            analysis.append("DETAIL LOG ANOMALI:")
            analysis.append("-" * 50)
            
            for pred in results["predictions"]:
                if pred["is_anomaly"]:
                    analysis.append(f"\n[{pred['index']}] Probability: {pred['probability']:.4f}")
                    analysis.append(f"    Log: {pred['log']}")
        
        return "\n".join(analysis)


def print_prediction_summary(results: Dict[str, Any]):
    """
    Print ringkasan prediksi.
    
    Args:
        results: Dictionary hasil prediksi
    """
    print("\n" + "=" * 60)
    print("HASIL DETEKSI ANOMALI")
    print("=" * 60)
    
    total = results["total_logs"]
    anomalies = results["anomaly_count"]
    normal = results["normal_count"]
    
    print(f"\n📊 Statistik:")
    print(f"   Total Log    : {total}")
    print(f"   Normal       : {normal} ({normal/total*100:.1f}%)")
    print(f"   Anomali      : {anomalies} ({anomalies/total*100:.1f}%)")
    
    if anomalies > 0:
        print(f"\n⚠️  DITEMUKAN {anomalies} LOG ANOMALI!")
        print("-" * 60)
        
        # Tampilkan anomali dengan probability tertinggi
        anomaly_logs = [p for p in results["predictions"] if p["is_anomaly"]]
        anomaly_logs.sort(key=lambda x: x["probability"], reverse=True)
        
        print("\nTop Anomali (berdasarkan probability):")
        for i, pred in enumerate(anomaly_logs[:10], 1):
            print(f"\n  {i}. [{pred['index']}] Prob: {pred['probability']:.4f}")
            print(f"     {pred['log']}")
    else:
        print("\n✅ Tidak ditemukan anomali!")


def interactive_mode(detector: AnomalyDetector):
    """
    Mode interaktif untuk testing.
    
    Args:
        detector: AnomalyDetector instance
    """
    print("\n" + "=" * 50)
    print("MODE INTERAKTIF")
    print("=" * 50)
    print("Masukkan log untuk dianalisis (ketik 'exit' untuk keluar)")
    print("-" * 50)
    
    while True:
        try:
            log = input("\n> ").strip()
            
            if log.lower() == 'exit':
                print("Keluar dari mode interaktif.")
                break
            
            if not log:
                continue
            
            # Prediksi
            result = detector.predict_single(log)
            
            # Tampilkan hasil
            status = "🔴 ANOMALI" if result["is_anomaly"] else "🟢 NORMAL"
            print(f"\nStatus: {status}")
            print(f"Probability: {result['probability']:.4f}")
            
        except KeyboardInterrupt:
            print("\n\nKeluar dari mode interaktif.")
            break


def batch_predict(detector: AnomalyDetector, input_dir: str, output_dir: str):
    """
    Prediksi batch pada direktori file log.
    
    Args:
        detector: AnomalyDetector instance
        input_dir: Direktori input
        output_dir: Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cari file log
    log_files = [f for f in os.listdir(input_dir) if f.endswith(('.log', '.txt'))]
    
    print(f"\nFound {len(log_files)} log files to process...")
    
    all_results = []
    
    for filename in log_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{filename}_predictions.json")
        
        print(f"\nProcessing {filename}...")
        results = detector.predict_file(input_path, output_path)
        
        all_results.append({
            "filename": filename,
            "total_logs": results["total_logs"],
            "anomaly_count": results["anomaly_count"]
        })
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN BATCH PREDICTION")
    print("=" * 60)
    
    total_logs = sum(r["total_logs"] for r in all_results)
    total_anomalies = sum(r["anomaly_count"] for r in all_results)
    
    print(f"\nTotal file diproses: {len(all_results)}")
    print(f"Total log: {total_logs}")
    print(f"Total anomali: {total_anomalies} ({total_anomalies/total_logs*100:.1f}%)")
    
    # Simpan ringkasan
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_files": len(all_results),
            "total_logs": total_logs,
            "total_anomalies": total_anomalies,
            "files": all_results
        }, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Detect anomalies in system logs')
    parser.add_argument('--model', type=str, default='models/saved_models/final_model.keras',
                       help='Path to trained model')
    parser.add_argument('--tokenizer', type=str, default='models/saved_models/tokenizer',
                       help='Path to tokenizer')
    parser.add_argument('--input', type=str, help='Input log file or directory')
    parser.add_argument('--output', type=str, help='Output file or directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--batch', action='store_true',
                       help='Process all files in input directory')
    
    args = parser.parse_args()
    
    # Inisialisasi detector
    detector = AnomalyDetector(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        threshold=args.threshold
    )
    
    if args.interactive:
        # Mode interaktif
        interactive_mode(detector)
        
    elif args.batch and args.input:
        # Batch processing
        output_dir = args.output or 'results/predictions'
        batch_predict(detector, args.input, output_dir)
        
    elif args.input:
        # Single file prediction
        if os.path.isfile(args.input):
            results = detector.predict_file(args.input, args.output)
            print_prediction_summary(results)
        else:
            print(f"Error: File tidak ditemukan: {args.input}")
            sys.exit(1)
    else:
        # Demo dengan sample logs
        print("\n" + "=" * 50)
        print("DEMO DETEKSI ANOMALI")
        print("=" * 50)
        
        sample_logs = [
            "Accepted password for user from 192.168.1.50 port 22 ssh2",
            "systemd[1]: Started nginx.service",
            "Failed password for invalid user admin from 185.220.101.1 port 22 ssh2",
            "Failed password for invalid user root from 185.220.101.1 port 22 ssh2",
            "kernel: Out of memory: Kill process 12345 (java) score 950",
            "[INFO] Application started successfully",
            "ALERT: Suspicious process detected: cryptominer (PID: 99999)",
            "Connection received: host=192.168.1.100 user=admin database=production",
        ]
        
        print("\nSample logs:")
        for i, log in enumerate(sample_logs):
            print(f"  [{i}] {log[:60]}...")
        
        results = detector.predict(sample_logs)
        print_prediction_summary(results)


if __name__ == "__main__":
    main()
