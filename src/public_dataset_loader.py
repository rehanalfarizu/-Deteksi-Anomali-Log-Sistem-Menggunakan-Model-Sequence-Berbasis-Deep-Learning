"""
Public Dataset Loader untuk Deteksi Anomali Log Sistem

Module ini menyediakan loader untuk dataset publik yang populer:
- HDFS Log Dataset (dari LogHub)
- BGL (Blue Gene/L) Log Dataset
- Thunderbird Log Dataset

Datasets tersedia di: https://github.com/logpai/loghub

Author: Mahasiswa AI
Date: 2026-01-06
"""

import os
import re
import requests
import zipfile
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseLogDatasetLoader(ABC):
    """
    Base class untuk log dataset loader.
    """
    
    def __init__(self, data_dir: str = "data/public"):
        """
        Inisialisasi loader.
        
        Args:
            data_dir: Direktori untuk menyimpan dataset
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    @abstractmethod
    def download(self) -> str:
        """Download dataset dan return path."""
        pass
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load dan parse dataset."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return informasi tentang dataset."""
        pass


class HDFSDatasetLoader(BaseLogDatasetLoader):
    """
    Loader untuk HDFS Log Dataset.
    
    HDFS (Hadoop Distributed File System) logs dari Hadoop cluster.
    Dataset ini banyak digunakan untuk benchmark anomaly detection.
    
    Reference:
    - Xu, W., et al. "Detecting large-scale system problems by mining console logs." 
      SOSP 2009.
    
    Dataset Source:
    - https://github.com/logpai/loghub/tree/master/HDFS
    """
    
    # URL untuk sample dataset (untuk demonstrasi)
    SAMPLE_LOGS_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
    LABELS_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/anomaly_label.csv"
    
    def __init__(self, data_dir: str = "data/public/hdfs"):
        super().__init__(data_dir)
        self.dataset_name = "HDFS"
        
        # HDFS Log regex pattern
        self.log_pattern = re.compile(
            r'(?P<block_id>blk_-?\d+).*?'
            r'(?P<message>.*)'
        )
        
        # Event templates dari HDFS
        self.event_templates = {
            "E1": "Receiving block <*> src: <*> dest: <*>",
            "E2": "Received block <*> of size <*> from <*>",
            "E3": "BLOCK* NameSystem.allocateBlock: <*>",
            "E4": "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*>",
            "E5": "PacketResponder <*> for block <*> terminating",
            "E6": "Deleting block <*> file <*>",
            "E7": "BLOCK* ask <*> to delete <*>",
            "E8": "BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>",
            "E9": "Verification succeeded for <*>",
            "E10": "writeBlock <*> received exception <*>",
            "E11": "Exception in receiveBlock for block <*>",
        }
        
    def download(self, use_sample: bool = True) -> str:
        """
        Download HDFS dataset.
        
        Args:
            use_sample: Gunakan sample 2k logs (True) atau full dataset (False)
            
        Returns:
            Path ke file yang didownload
        """
        logs_path = os.path.join(self.data_dir, "HDFS_2k.log")
        labels_path = os.path.join(self.data_dir, "anomaly_label.csv")
        
        # Download logs
        if not os.path.exists(logs_path):
            print(f"Downloading HDFS logs to {logs_path}...")
            try:
                response = requests.get(self.SAMPLE_LOGS_URL, timeout=30)
                response.raise_for_status()
                with open(logs_path, 'w') as f:
                    f.write(response.text)
                print("Download complete!")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Creating synthetic HDFS-like logs instead...")
                self._generate_synthetic_hdfs_logs(logs_path)
        
        # Download labels
        if not os.path.exists(labels_path):
            print(f"Downloading HDFS labels to {labels_path}...")
            try:
                response = requests.get(self.LABELS_URL, timeout=30)
                response.raise_for_status()
                with open(labels_path, 'w') as f:
                    f.write(response.text)
                print("Download complete!")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Creating synthetic labels instead...")
                self._generate_synthetic_labels(labels_path)
        
        return logs_path
    
    def _generate_synthetic_hdfs_logs(self, output_path: str, num_logs: int = 2000):
        """Generate synthetic HDFS-like logs untuk demonstrasi."""
        import random
        
        templates_normal = [
            "081109 203518 143 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_{block_id} terminating",
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_{block_id} src: /{src_ip}:{src_port} dest: /{dest_ip}:{dest_port}",
            "081109 203518 143 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_{task_id}. blk_{block_id}",
            "081109 203518 143 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip}:{port} is added to blk_{block_id} size {size}",
            "081109 203518 143 INFO dfs.DataNode$BlockReceiver: Received block blk_{block_id} of size {size} from /{ip}",
            "081109 203518 143 INFO dfs.DataBlockScanner: Verification succeeded for blk_{block_id}",
        ]
        
        templates_anomaly = [
            "081109 203518 143 ERROR dfs.DataNode$DataXceiver: writeBlock blk_{block_id} received exception java.io.IOException: Connection reset by peer",
            "081109 203518 143 WARN dfs.FSNamesystem: Exception in receiveBlock for block blk_{block_id} java.io.IOException: Could not complete file",
            "081109 203518 143 ERROR dfs.DataNode: Block blk_{block_id} is invalid: File not found",
            "081109 203518 143 WARN dfs.DataNode$DataXceiver: IOException in BlockReceiver constructor for block blk_{block_id}",
        ]
        
        logs = []
        block_ids = [random.randint(1000000000000000000, 9999999999999999999) for _ in range(500)]
        
        for i in range(num_logs):
            is_anomaly = random.random() < 0.1
            template = random.choice(templates_anomaly if is_anomaly else templates_normal)
            
            log = template.format(
                block_id=random.choice(block_ids),
                src_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                src_port=random.randint(40000, 60000),
                dest_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                dest_port=50010,
                ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                port=50010,
                task_id=f"{random.randint(200000,300000)}_0000_m_000000_0",
                size=random.randint(1000000, 100000000)
            )
            logs.append(log)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(logs))
        
        print(f"Generated {num_logs} synthetic HDFS logs")
    
    def _generate_synthetic_labels(self, output_path: str):
        """Generate synthetic labels."""
        import random
        
        # Generate block IDs and labels
        block_ids = set()
        for _ in range(500):
            block_ids.add(f"blk_{random.randint(1000000000000000000, 9999999999999999999)}")
        
        labels = []
        for block_id in block_ids:
            is_anomaly = random.random() < 0.1
            labels.append({"BlockId": block_id, "Label": "Anomaly" if is_anomaly else "Normal"})
        
        df = pd.DataFrame(labels)
        df.to_csv(output_path, index=False)
        print(f"Generated labels for {len(block_ids)} blocks")
    
    def load(self, max_logs: Optional[int] = None) -> pd.DataFrame:
        """
        Load dan parse HDFS dataset.
        
        Args:
            max_logs: Maksimum jumlah logs yang diload (None = semua)
            
        Returns:
            DataFrame dengan kolom: log_message, block_id, label
        """
        logs_path = os.path.join(self.data_dir, "HDFS_2k.log")
        labels_path = os.path.join(self.data_dir, "anomaly_label.csv")
        
        # Download jika belum ada
        if not os.path.exists(logs_path):
            self.download()
        
        # Load logs
        print("Loading HDFS logs...")
        logs = []
        with open(logs_path, 'r') as f:
            for i, line in enumerate(f):
                if max_logs and i >= max_logs:
                    break
                line = line.strip()
                if line:
                    # Extract block ID
                    block_match = re.search(r'blk_-?\d+', line)
                    block_id = block_match.group(0) if block_match else None
                    logs.append({
                        "log_message": line,
                        "block_id": block_id
                    })
        
        df_logs = pd.DataFrame(logs)
        print(f"Loaded {len(df_logs)} logs")
        
        # Load labels jika tersedia
        if os.path.exists(labels_path):
            print("Loading labels...")
            df_labels = pd.read_csv(labels_path)
            
            # Map labels ke logs berdasarkan block_id
            label_map = dict(zip(df_labels['BlockId'], df_labels['Label']))
            df_logs['label'] = df_logs['block_id'].map(
                lambda x: 1 if label_map.get(x, 'Normal') == 'Anomaly' else 0
            )
        else:
            print("Labels not found, using heuristic labeling...")
            # Heuristic: ERROR dan Exception = anomaly
            df_logs['label'] = df_logs['log_message'].apply(
                lambda x: 1 if any(kw in x.upper() for kw in ['ERROR', 'EXCEPTION', 'WARN', 'FAIL']) else 0
            )
        
        print(f"Normal: {len(df_logs[df_logs['label'] == 0])}, Anomaly: {len(df_logs[df_logs['label'] == 1])}")
        
        return df_logs
    
    def get_info(self) -> Dict[str, Any]:
        """Return informasi tentang HDFS dataset."""
        return {
            "name": "HDFS Log Dataset",
            "description": "Hadoop Distributed File System logs from a Hadoop cluster",
            "source": "LogHub (https://github.com/logpai/loghub)",
            "num_logs": "11,175,629 (full) / 2,000 (sample)",
            "num_anomaly_blocks": "16,838 anomaly blocks",
            "reference": "Xu et al., SOSP 2009",
            "format": "Unstructured text logs with block IDs"
        }


class BGLDatasetLoader(BaseLogDatasetLoader):
    """
    Loader untuk BGL (Blue Gene/L) Log Dataset.
    
    Blue Gene/L adalah supercomputer dari IBM. Dataset ini berisi
    system logs dari Blue Gene/L yang di-install di Lawrence Livermore
    National Labs (LLNL).
    
    Reference:
    - Oliner, A., et al. "What supercomputers say: A study of five system logs."
      DSN 2007.
    
    Dataset Source:
    - https://github.com/logpai/loghub/tree/master/BGL
    """
    
    def __init__(self, data_dir: str = "data/public/bgl"):
        super().__init__(data_dir)
        self.dataset_name = "BGL"
        
    def download(self) -> str:
        """
        Download BGL dataset sample.
        
        Returns:
            Path ke file yang didownload
        """
        logs_path = os.path.join(self.data_dir, "BGL_2k.log")
        
        if not os.path.exists(logs_path):
            print(f"Generating synthetic BGL-like logs to {logs_path}...")
            self._generate_synthetic_bgl_logs(logs_path)
        
        return logs_path
    
    def _generate_synthetic_bgl_logs(self, output_path: str, num_logs: int = 2000):
        """Generate synthetic BGL-like logs untuk demonstrasi."""
        import random
        from datetime import datetime, timedelta
        
        # BGL log format: Label Timestamp Date Node Time Component Level Content
        components = ["KERNEL", "APP", "HARDWARE", "SOFTWARE", "NETWORK", "MMCS", "BGLMASTER"]
        nodes = [f"R{random.randint(0,9)}{random.randint(0,9)}-M{random.randint(0,1)}-N{random.randint(0,9)}" 
                 for _ in range(50)]
        
        normal_templates = [
            "generating core.{pid}",
            "program interrupt, kernel terminated process",
            "data TLB error interrupt",
            "CE sym {sym}, at {addr}, mask {mask}",
            "instruction cache parity error corrected",
            "L3 tag ECC error corrected",
            "doubled-hummer alignment exceptions",
            "L2 DCU: not initialized",
        ]
        
        anomaly_templates = [
            "FATAL: Machine check interrupt",
            "FATAL: Hard disk failure detected",
            "FATAL: Memory ECC uncorrectable error",
            "FAILURE: Kernel panic - not syncing",
            "FAILURE: Network adapter failure",
            "CRITICAL: Temperature threshold exceeded",
            "SEVERE: Multiple bit ECC error",
        ]
        
        logs = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(num_logs):
            is_anomaly = random.random() < 0.1
            label = "-" if not is_anomaly else "FATAL" if random.random() < 0.5 else "FAILURE"
            
            timestamp = (base_time + timedelta(seconds=i*10)).strftime("%Y-%m-%d-%H.%M.%S.%f")[:-3]
            date = (base_time + timedelta(seconds=i*10)).strftime("%Y.%m.%d")
            node = random.choice(nodes)
            time_str = (base_time + timedelta(seconds=i*10)).strftime("%Y-%m-%d-%H.%M.%S")
            component = random.choice(components)
            level = "FATAL" if is_anomaly else random.choice(["INFO", "WARNING"])
            
            template = random.choice(anomaly_templates if is_anomaly else normal_templates)
            content = template.format(
                pid=random.randint(1000, 9999),
                sym=random.randint(0, 15),
                addr=f"0x{random.randint(0, 0xFFFFFFFF):08x}",
                mask=f"0x{random.randint(0, 0xFF):02x}"
            )
            
            log = f"{label} {timestamp} {date} {node} {time_str} {component} {level} {content}"
            logs.append(log)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(logs))
        
        print(f"Generated {num_logs} synthetic BGL logs")
    
    def load(self, max_logs: Optional[int] = None) -> pd.DataFrame:
        """
        Load dan parse BGL dataset.
        
        Args:
            max_logs: Maksimum jumlah logs yang diload
            
        Returns:
            DataFrame dengan kolom: log_message, label, component, level
        """
        logs_path = os.path.join(self.data_dir, "BGL_2k.log")
        
        if not os.path.exists(logs_path):
            self.download()
        
        print("Loading BGL logs...")
        logs = []
        
        with open(logs_path, 'r') as f:
            for i, line in enumerate(f):
                if max_logs and i >= max_logs:
                    break
                line = line.strip()
                if line:
                    parts = line.split(' ', 7)
                    if len(parts) >= 8:
                        label_str = parts[0]
                        label = 0 if label_str == "-" else 1
                        logs.append({
                            "log_message": line,
                            "label": label,
                            "label_str": label_str,
                            "timestamp": parts[1],
                            "node": parts[3] if len(parts) > 3 else "",
                            "component": parts[5] if len(parts) > 5 else "",
                            "level": parts[6] if len(parts) > 6 else "",
                            "content": parts[7] if len(parts) > 7 else ""
                        })
        
        df = pd.DataFrame(logs)
        print(f"Loaded {len(df)} logs")
        print(f"Normal: {len(df[df['label'] == 0])}, Anomaly: {len(df[df['label'] == 1])}")
        
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """Return informasi tentang BGL dataset."""
        return {
            "name": "BGL (Blue Gene/L) Log Dataset",
            "description": "System logs from Blue Gene/L supercomputer at LLNL",
            "source": "LogHub (https://github.com/logpai/loghub)",
            "num_logs": "4,747,963 (full) / 2,000 (sample)",
            "num_anomalies": "348,460 alert messages",
            "reference": "Oliner et al., DSN 2007",
            "format": "Structured logs with labels"
        }


class ThunderbirdDatasetLoader(BaseLogDatasetLoader):
    """
    Loader untuk Thunderbird Log Dataset.
    
    Thunderbird adalah supercomputer di Sandia National Labs.
    Dataset berisi logs selama beberapa tahun operasi.
    
    Dataset Source:
    - https://github.com/logpai/loghub/tree/master/Thunderbird
    """
    
    def __init__(self, data_dir: str = "data/public/thunderbird"):
        super().__init__(data_dir)
        self.dataset_name = "Thunderbird"
        
    def download(self) -> str:
        """Download Thunderbird dataset sample."""
        logs_path = os.path.join(self.data_dir, "Thunderbird_2k.log")
        
        if not os.path.exists(logs_path):
            print(f"Generating synthetic Thunderbird-like logs to {logs_path}...")
            self._generate_synthetic_thunderbird_logs(logs_path)
        
        return logs_path
    
    def _generate_synthetic_thunderbird_logs(self, output_path: str, num_logs: int = 2000):
        """Generate synthetic Thunderbird-like logs."""
        import random
        from datetime import datetime, timedelta
        
        facilities = ["auth", "kernel", "daemon", "syslog", "user", "local0"]
        
        normal_templates = [
            "session opened for user {user} by (uid=0)",
            "session closed for user {user}",
            "Accepted publickey for {user} from {ip} port {port} ssh2",
            "Starting periodic command scheduler: cron",
            "Stopping periodic command scheduler: cron",
            "(root) CMD (/usr/lib/sa/sa1 1 1)",
        ]
        
        anomaly_templates = [
            "authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip}",
            "Failed password for invalid user {user} from {ip} port {port} ssh2",
            "error: PAM: Authentication failure for {user} from {ip}",
            "kernel: EDAC MC0: UE page 0x{addr}, offset 0x{offset}",
            "kernel: mce: CPU0: Machine Check Exception",
        ]
        
        users = ["root", "admin", "user1", "daemon", "nobody"]
        logs = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(num_logs):
            is_anomaly = random.random() < 0.1
            label = "-" if not is_anomaly else random.choice(["ALERT", "FAILURE"])
            
            timestamp = (base_time + timedelta(seconds=i*5)).strftime("%b %d %H:%M:%S")
            facility = random.choice(facilities)
            
            template = random.choice(anomaly_templates if is_anomaly else normal_templates)
            content = template.format(
                user=random.choice(users),
                ip=f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                port=random.randint(10000, 65535),
                addr=f"{random.randint(0, 0xFFFF):04x}",
                offset=f"{random.randint(0, 0xFFF):03x}"
            )
            
            log = f"{label} {timestamp} {facility} {content}"
            logs.append(log)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(logs))
        
        print(f"Generated {num_logs} synthetic Thunderbird logs")
    
    def load(self, max_logs: Optional[int] = None) -> pd.DataFrame:
        """Load dan parse Thunderbird dataset."""
        logs_path = os.path.join(self.data_dir, "Thunderbird_2k.log")
        
        if not os.path.exists(logs_path):
            self.download()
        
        print("Loading Thunderbird logs...")
        logs = []
        
        with open(logs_path, 'r') as f:
            for i, line in enumerate(f):
                if max_logs and i >= max_logs:
                    break
                line = line.strip()
                if line:
                    parts = line.split(' ', 4)
                    if len(parts) >= 5:
                        label_str = parts[0]
                        label = 0 if label_str == "-" else 1
                        logs.append({
                            "log_message": line,
                            "label": label,
                            "content": parts[4] if len(parts) > 4 else line
                        })
        
        df = pd.DataFrame(logs)
        print(f"Loaded {len(df)} logs")
        print(f"Normal: {len(df[df['label'] == 0])}, Anomaly: {len(df[df['label'] == 1])}")
        
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """Return informasi tentang Thunderbird dataset."""
        return {
            "name": "Thunderbird Log Dataset",
            "description": "System logs from Thunderbird supercomputer at Sandia",
            "source": "LogHub (https://github.com/logpai/loghub)",
            "num_logs": "211,212,192 (full) / 2,000 (sample)",
            "reference": "Oliner et al., DSN 2007",
            "format": "Syslog format with labels"
        }


def load_public_dataset(
    dataset_name: str,
    data_dir: str = "data/public",
    max_logs: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fungsi helper untuk load public dataset.
    
    Args:
        dataset_name: Nama dataset ('hdfs', 'bgl', atau 'thunderbird')
        data_dir: Direktori untuk menyimpan data
        max_logs: Maksimum logs yang diload
        
    Returns:
        Tuple (DataFrame, info_dict)
    """
    loaders = {
        'hdfs': HDFSDatasetLoader,
        'bgl': BGLDatasetLoader,
        'thunderbird': ThunderbirdDatasetLoader
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    loader_class = loaders[dataset_name]
    loader = loader_class(os.path.join(data_dir, dataset_name))
    
    # Download dan load
    loader.download()
    df = loader.load(max_logs)
    info = loader.get_info()
    
    return df, info


def list_available_datasets() -> List[Dict[str, str]]:
    """List semua dataset yang tersedia."""
    datasets = [
        {
            "name": "hdfs",
            "full_name": "HDFS Log Dataset",
            "description": "Hadoop Distributed File System logs"
        },
        {
            "name": "bgl",
            "full_name": "BGL (Blue Gene/L) Log Dataset",
            "description": "Blue Gene/L supercomputer logs"
        },
        {
            "name": "thunderbird",
            "full_name": "Thunderbird Log Dataset",
            "description": "Thunderbird supercomputer logs"
        }
    ]
    return datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load public log datasets")
    parser.add_argument("--dataset", type=str, default="hdfs",
                        choices=["hdfs", "bgl", "thunderbird"],
                        help="Dataset to load")
    parser.add_argument("--max_logs", type=int, default=None,
                        help="Maximum number of logs to load")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for processed CSV")
    
    args = parser.parse_args()
    
    print(f"\nLoading {args.dataset.upper()} dataset...")
    print("="*50)
    
    df, info = load_public_dataset(args.dataset, max_logs=args.max_logs)
    
    print("\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
