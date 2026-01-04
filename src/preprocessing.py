"""
Preprocessing Module untuk Deteksi Anomali Log Sistem

Module ini berisi fungsi-fungsi untuk preprocessing data log
sebelum digunakan untuk training model deep learning.

Author: Mahasiswa AI
Date: 2026-01-04
"""

import re
import string
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from collections import Counter


class LogPreprocessor:
    """
    Preprocessor untuk data log sistem.
    
    Attributes:
        lowercase (bool): Konversi ke lowercase
        remove_special_chars (bool): Hapus karakter spesial
        remove_numbers (bool): Hapus angka
        min_word_freq (int): Frekuensi minimum kata
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        min_word_freq: int = 2
    ):
        """
        Inisialisasi preprocessor.
        
        Args:
            lowercase: Konversi teks ke lowercase
            remove_special_chars: Hapus karakter spesial
            remove_numbers: Hapus angka dari teks
            min_word_freq: Frekuensi minimum untuk kata yang dipertahankan
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.min_word_freq = min_word_freq
        
        # Regex patterns
        self.ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        self.timestamp_pattern = re.compile(
            r'\d{4}-\d{2}-\d{2}[\s|T]\d{2}:\d{2}:\d{2}|\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}'
        )
        self.port_pattern = re.compile(r'port\s+\d+')
        self.pid_pattern = re.compile(r'\[\d+\]|pid\s*[=:]\s*\d+', re.IGNORECASE)
        self.path_pattern = re.compile(r'/[\w\-./]+')
        self.hex_pattern = re.compile(r'0x[0-9a-fA-F]+')
        self.number_pattern = re.compile(r'\b\d+\b')
        
        # Placeholder tokens
        self.PLACEHOLDER_IP = "<IP>"
        self.PLACEHOLDER_TIMESTAMP = "<TIMESTAMP>"
        self.PLACEHOLDER_PORT = "<PORT>"
        self.PLACEHOLDER_PID = "<PID>"
        self.PLACEHOLDER_PATH = "<PATH>"
        self.PLACEHOLDER_HEX = "<HEX>"
        self.PLACEHOLDER_NUM = "<NUM>"
        
    def normalize_ip(self, text: str) -> str:
        """Normalisasi alamat IP."""
        return self.ip_pattern.sub(self.PLACEHOLDER_IP, text)
    
    def normalize_timestamp(self, text: str) -> str:
        """Normalisasi timestamp."""
        return self.timestamp_pattern.sub(self.PLACEHOLDER_TIMESTAMP, text)
    
    def normalize_port(self, text: str) -> str:
        """Normalisasi nomor port."""
        return self.port_pattern.sub(f"port {self.PLACEHOLDER_PORT}", text)
    
    def normalize_pid(self, text: str) -> str:
        """Normalisasi Process ID."""
        return self.pid_pattern.sub(self.PLACEHOLDER_PID, text)
    
    def normalize_path(self, text: str) -> str:
        """Normalisasi file path."""
        return self.path_pattern.sub(self.PLACEHOLDER_PATH, text)
    
    def normalize_hex(self, text: str) -> str:
        """Normalisasi nilai hexadecimal."""
        return self.hex_pattern.sub(self.PLACEHOLDER_HEX, text)
    
    def normalize_numbers(self, text: str) -> str:
        """Normalisasi angka."""
        return self.number_pattern.sub(self.PLACEHOLDER_NUM, text)
    
    def clean_text(self, text: str) -> str:
        """
        Bersihkan teks dari karakter yang tidak diperlukan.
        
        Args:
            text: Teks input
            
        Returns:
            Teks yang sudah dibersihkan
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalisasi
        text = self.normalize_ip(text)
        text = self.normalize_timestamp(text)
        text = self.normalize_port(text)
        text = self.normalize_pid(text)
        text = self.normalize_hex(text)
        
        # Hapus karakter spesial (kecuali placeholder)
        if self.remove_special_chars:
            # Pertahankan huruf, angka, spasi, dan placeholder
            text = re.sub(r'[^\w\s<>]', ' ', text)
        
        # Hapus angka
        if self.remove_numbers:
            text = self.normalize_numbers(text)
        
        # Hapus multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, logs: List[str]) -> List[str]:
        """
        Preprocess daftar log.
        
        Args:
            logs: List log messages
            
        Returns:
            List log yang sudah dipreprocess
        """
        return [self.clean_text(log) for log in logs]
    
    def extract_log_components(self, log: str) -> Dict[str, str]:
        """
        Ekstrak komponen dari log message.
        
        Args:
            log: Log message
            
        Returns:
            Dictionary berisi komponen log
        """
        components = {
            "timestamp": None,
            "ip_addresses": [],
            "ports": [],
            "paths": [],
            "message": log
        }
        
        # Ekstrak timestamp
        timestamp_match = self.timestamp_pattern.search(log)
        if timestamp_match:
            components["timestamp"] = timestamp_match.group()
        
        # Ekstrak IP addresses
        components["ip_addresses"] = self.ip_pattern.findall(log)
        
        # Ekstrak ports
        port_matches = re.findall(r'port\s+(\d+)', log, re.IGNORECASE)
        components["ports"] = port_matches
        
        # Ekstrak paths
        components["paths"] = self.path_pattern.findall(log)
        
        return components


class LogParser:
    """
    Parser untuk berbagai format log.
    """
    
    def __init__(self):
        """Inisialisasi parser."""
        # Pattern untuk berbagai format log
        self.syslog_pattern = re.compile(
            r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'
            r'(\S+)\s+'
            r'(\S+?)(?:\[(\d+)\])?:\s+'
            r'(.*)$'
        )
        
        self.apache_pattern = re.compile(
            r'^(\S+)\s+-\s+-\s+'
            r'\[([^\]]+)\]\s+'
            r'"(\w+)\s+(\S+)\s+(\S+)"\s+'
            r'(\d+)\s+(\d+)'
        )
        
        self.json_pattern = re.compile(
            r'\{[^}]+\}'
        )
        
    def parse_syslog(self, log: str) -> Optional[Dict]:
        """
        Parse format syslog.
        
        Args:
            log: Log message dalam format syslog
            
        Returns:
            Dictionary dengan komponen log atau None
        """
        match = self.syslog_pattern.match(log)
        if match:
            return {
                "timestamp": match.group(1),
                "hostname": match.group(2),
                "program": match.group(3),
                "pid": match.group(4),
                "message": match.group(5)
            }
        return None
    
    def parse_apache(self, log: str) -> Optional[Dict]:
        """
        Parse format Apache access log.
        
        Args:
            log: Log message dalam format Apache
            
        Returns:
            Dictionary dengan komponen log atau None
        """
        match = self.apache_pattern.match(log)
        if match:
            return {
                "ip": match.group(1),
                "timestamp": match.group(2),
                "method": match.group(3),
                "path": match.group(4),
                "protocol": match.group(5),
                "status": match.group(6),
                "bytes": match.group(7)
            }
        return None
    
    def detect_format(self, log: str) -> str:
        """
        Deteksi format log.
        
        Args:
            log: Log message
            
        Returns:
            Nama format yang terdeteksi
        """
        if self.syslog_pattern.match(log):
            return "syslog"
        elif self.apache_pattern.match(log):
            return "apache"
        elif self.json_pattern.search(log):
            return "json"
        else:
            return "unknown"
    
    def parse(self, log: str) -> Dict:
        """
        Parse log secara otomatis berdasarkan format.
        
        Args:
            log: Log message
            
        Returns:
            Dictionary dengan komponen log
        """
        format_type = self.detect_format(log)
        
        if format_type == "syslog":
            result = self.parse_syslog(log)
        elif format_type == "apache":
            result = self.parse_apache(log)
        else:
            result = {"raw": log}
        
        if result:
            result["format"] = format_type
        
        return result


class FeatureExtractor:
    """
    Ekstrak fitur dari log untuk model.
    """
    
    def __init__(self):
        """Inisialisasi extractor."""
        self.keyword_patterns = {
            "error": re.compile(r'\b(error|err|fail|failed|failure|exception)\b', re.IGNORECASE),
            "warning": re.compile(r'\b(warn|warning|alert)\b', re.IGNORECASE),
            "critical": re.compile(r'\b(critical|crit|fatal|emergency)\b', re.IGNORECASE),
            "success": re.compile(r'\b(success|succeeded|accepted|ok)\b', re.IGNORECASE),
            "authentication": re.compile(r'\b(auth|login|password|user|credential)\b', re.IGNORECASE),
            "network": re.compile(r'\b(connection|connect|disconnect|port|ip|network)\b', re.IGNORECASE),
            "file": re.compile(r'\b(file|read|write|open|close|delete|create)\b', re.IGNORECASE),
        }
        
    def extract_statistical_features(self, log: str) -> Dict[str, float]:
        """
        Ekstrak fitur statistik dari log.
        
        Args:
            log: Log message
            
        Returns:
            Dictionary fitur statistik
        """
        features = {
            "length": len(log),
            "word_count": len(log.split()),
            "digit_ratio": sum(c.isdigit() for c in log) / max(len(log), 1),
            "upper_ratio": sum(c.isupper() for c in log) / max(len(log), 1),
            "special_char_ratio": sum(not c.isalnum() and not c.isspace() for c in log) / max(len(log), 1),
        }
        
        return features
    
    def extract_keyword_features(self, log: str) -> Dict[str, int]:
        """
        Ekstrak fitur berdasarkan keyword.
        
        Args:
            log: Log message
            
        Returns:
            Dictionary fitur keyword (0 atau 1)
        """
        features = {}
        
        for keyword, pattern in self.keyword_patterns.items():
            features[f"has_{keyword}"] = 1 if pattern.search(log) else 0
        
        return features
    
    def extract_all_features(self, log: str) -> Dict[str, float]:
        """
        Ekstrak semua fitur dari log.
        
        Args:
            log: Log message
            
        Returns:
            Dictionary semua fitur
        """
        features = {}
        features.update(self.extract_statistical_features(log))
        features.update(self.extract_keyword_features(log))
        
        return features


def preprocess_dataframe(
    df: pd.DataFrame,
    log_column: str = "log_message",
    preprocessor: LogPreprocessor = None
) -> pd.DataFrame:
    """
    Preprocess DataFrame yang berisi log.
    
    Args:
        df: DataFrame dengan log
        log_column: Nama kolom yang berisi log
        preprocessor: Instance LogPreprocessor (opsional)
        
    Returns:
        DataFrame dengan kolom log yang sudah dipreprocess
    """
    if preprocessor is None:
        preprocessor = LogPreprocessor()
    
    df = df.copy()
    df["processed_log"] = df[log_column].apply(preprocessor.clean_text)
    
    return df


def create_vocabulary(
    logs: List[str],
    max_words: int = 10000,
    min_freq: int = 2
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Buat vocabulary dari daftar log.
    
    Args:
        logs: List log messages
        max_words: Jumlah maksimum kata dalam vocabulary
        min_freq: Frekuensi minimum kata
        
    Returns:
        Tuple (word_to_idx, idx_to_word)
    """
    # Hitung frekuensi kata
    word_counts = Counter()
    for log in logs:
        words = log.split()
        word_counts.update(words)
    
    # Filter berdasarkan frekuensi minimum
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Ambil top N kata
    most_common = word_counts.most_common(max_words)
    
    # Buat mapping
    word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    for idx, (word, _) in enumerate(most_common, start=4):
        if word not in word_to_idx:
            word_to_idx[word] = idx
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


if __name__ == "__main__":
    # Contoh penggunaan
    sample_logs = [
        "Jan  4 10:15:23 server01 sshd[12345]: Failed password for invalid user admin from 192.168.1.100 port 22 ssh2",
        "192.168.1.100 - - [04/Jan/2026:10:15:23 +0700] \"GET /admin HTTP/1.1\" 404 512",
        "2026-01-04 10:15:23 ERROR [ModuleName] Database connection failed",
    ]
    
    # Test preprocessor
    preprocessor = LogPreprocessor()
    for log in sample_logs:
        processed = preprocessor.clean_text(log)
        print(f"Original: {log[:50]}...")
        print(f"Processed: {processed[:50]}...")
        print()
    
    # Test parser
    parser = LogParser()
    for log in sample_logs:
        parsed = parser.parse(log)
        print(f"Format: {parsed.get('format', 'unknown')}")
        print(f"Parsed: {parsed}")
        print()
    
    # Test feature extractor
    extractor = FeatureExtractor()
    for log in sample_logs:
        features = extractor.extract_all_features(log)
        print(f"Features for: {log[:30]}...")
        print(f"Features: {features}")
        print()
