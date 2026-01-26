# 🚀 Setup Google Colab Badge

## Cara Update Link Colab di README

Setelah Anda push repository ke GitHub, update badge Colab di `README.md` dengan langkah berikut:

### 1. Identifikasi Informasi Repository Anda

- **GitHub Username**: (contoh: `john-doe`)
- **Repository Name**: `Deteksi-Anomali-Log-Sistem-Menggunakan-Model-Sequence-Berbasis-Deep-Learning`
- **Branch**: `main` (atau `master`)

### 2. Format URL Colab

```
https://colab.research.google.com/github/{USERNAME}/{REPO_NAME}/blob/{BRANCH}/notebooks/Anomaly_Detection_Colab.ipynb
```

### 3. Contoh Lengkap

Jika username GitHub Anda adalah `john-doe`, ganti di README.md:

**Dari:**
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Deteksi-Anomali-Log-Sistem-Menggunakan-Model-Sequence-Berbasis-Deep-Learning/blob/main/notebooks/Anomaly_Detection_Colab.ipynb)
```

**Menjadi:**
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/john-doe/Deteksi-Anomali-Log-Sistem-Menggunakan-Model-Sequence-Berbasis-Deep-Learning/blob/main/notebooks/Anomaly_Detection_Colab.ipynb)
```

### 4. Lokasi yang Perlu Diganti

Ada **2 tempat** di README.md yang perlu di-update:

1. **Baris ~6** - Badge di bagian atas
2. **Baris ~120** - Badge di section "Google Colab (Recommended)"
3. **Baris ~322** - Badge di section "Jupyter Notebook"

### 5. Cara Cepat dengan Find & Replace

Di editor Anda (VS Code), gunakan Find & Replace:

- **Find**: `yourusername`
- **Replace**: `john-doe` (ganti dengan username GitHub Anda)
- **Replace All**

### 6. Verifikasi

Setelah push ke GitHub:

1. Buka README.md di GitHub
2. Klik badge "Open in Colab"
3. Pastikan notebook terbuka di Google Colab
4. Jika error 404, cek kembali username dan branch name

## Troubleshooting

### ❌ Error: "Notebook not found"

**Penyebab:**
- Username salah
- Branch name salah (main vs master)
- Repository masih private
- Path notebook salah

**Solusi:**
1. Pastikan repository adalah **public** di GitHub
2. Cek branch name dengan: `git branch --show-current`
3. Verifikasi path: `notebooks/Anomaly_Detection_Colab.ipynb`

### ❌ Badge tidak klik-able

**Penyebab:** Syntax markdown salah

**Format yang benar:**
```markdown
[![Text](badge-image-url)](target-url)
```

## Test Link Secara Manual

Buka browser dan test URL ini (ganti `{USERNAME}`):

```
https://colab.research.google.com/github/{USERNAME}/Deteksi-Anomali-Log-Sistem-Menggunakan-Model-Sequence-Berbasis-Deep-Learning/blob/main/notebooks/Anomaly_Detection_Colab.ipynb
```

Jika terbuka dengan baik, copy URL tersebut ke badge di README.md.

## Shortcut URL Colab

Alternatif cara membuka notebook di Colab:

1. Buka: https://colab.research.google.com
2. Klik tab "GitHub"  
3. Paste URL repository Anda
4. Pilih notebook `Anomaly_Detection_Colab.ipynb`
5. Copy URL dari address bar

---

✅ Setelah setup selesai, user lain bisa langsung klik badge untuk membuka notebook di Colab!
