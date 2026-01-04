# Data Directory

Direktori ini berisi data log untuk training dan testing model deteksi anomali.

## Struktur

```
data/
├── raw/                    # Data log mentah
│   └── sample_logs.txt     # Contoh log
└── processed/              # Data yang sudah diproses
    ├── train_logs.csv      # Data training
    └── test_logs.csv       # Data testing
```

## Generate Data

Untuk generate data sintetis, jalankan:

```bash
python src/data_generator.py --num_logs 10000 --anomaly_ratio 0.1
```
