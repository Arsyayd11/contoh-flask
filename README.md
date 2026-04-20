# Contoh Flask — Deployment Model Machine Learning

Repositori ini berisi empat contoh aplikasi web berbasis Flask yang mendemonstrasikan
deployment berbagai jenis model machine learning, dilengkapi dengan penjelasan kode
yang mudah dipahami mahasiswa.

---

## Struktur Folder

```
contoh-flask/
│
├── app.py                  ← Versi SEDERHANA (penjelasan dasar Flask + Decision Tree)
├── model.joblib            ← Model Decision Tree terlatih + akurasi
├── train_simple.py         ← Script training versi sederhana
├── templates/
│   └── index.html          ← Form prediksi + confidence score + akurasi
│
├── klasifikasi/            ← HeartCheck: Heart Disease · Tema Kesehatan Hijau-Putih
│   ├── app.py              ← Semua route + auth session + simpan riwayat
│   ├── database.py         ← SQLite helper: init, register, login, riwayat
│   ├── heartcheck.db       ← Database SQLite (dibuat otomatis saat run pertama)
│   ├── model.joblib        ← Pipeline terlatih + metadata (accuracy, AUC, importances)
│   ├── train.py            ← Training profesional dengan CV + evaluasi lengkap
│   ├── heart_vailure_dataset.csv ← Dataset Heart Failure (918 observasi, 11 fitur)
│   └── templates/
│       ├── base.html       ← Base template: navbar + dropdown profil (JS click-toggle) + Riwayat link kondisional
│       ├── landing.html    ← Hero landing page: statistik model + cara kerja + CTA
│       ├── prediksi.html   ← Form 1-kolom + tooltip ⓘ + 6 profil contoh acak + hasil di bawah form
│       ├── login.html      ← Halaman masuk akun
│       ├── register.html   ← Halaman daftar akun baru (validasi real-time)
│       ├── riwayat.html    ← Riwayat prediksi sebagai accordion card (expandable, harus login)
│       └── tentang.html    ← Tentang aplikasi, dataset, model, FAQ, disclaimer
│
├── regresi/                ← Boston Housing + Gradient Boosting + interval kepercayaan
│   ├── app.py              ← Route: / (landing) + /prediksi
│   ├── model.joblib        ← Model terlatih + R², MAE
│   ├── train.py            ← Training dengan evaluasi R² dan MAE
│   └── templates/
│       ├── landing.html    ← Halaman landing: penjelasan + performa model
│       └── index.html      ← Form + harga estimasi + interval ±MAE + R² bar
│
├── arima/                  ← Whoosh Jakarta–Bandung + ARIMA + Chart.js
│   ├── app.py              ← Dashboard prediksi penumpang Whoosh
│   ├── model.joblib        ← ARIMA model + series historis
│   ├── train.py            ← Data sintetis penumpang Whoosh + ARIMA(1,1,1)
│   └── templates/
│       └── index.html      ← Chart historis + forecast tabel + statistik
│
└── venv/                   ← Virtual environment (dibuat manual, lihat di bawah)
```

---

## Penjelasan Singkat Tiap Aplikasi

### 1. Versi Sederhana (root) — Decision Tree Classifier
- **Dataset**: Titanic (200 baris inline)
- **Model**: Decision Tree (max_depth=4)
- **Fitur**: Pclass, Sex, Age, Fare
- **Tambahan**: Confidence score (predict_proba) + tampilan akurasi model
- **Cocok untuk**: Memahami alur dasar Flask → form → prediksi → tampil hasil

### 2. Klasifikasi — HeartCheck (Heart Disease Predictor)
- **Dataset**: Heart Failure Prediction (918 observasi, 11 fitur klinis)
- **Model**: Random Forest Classifier dengan Pipeline preprocessing (StandardScaler + OneHotEncoder)
- **Evaluasi**: Akurasi, AUC-ROC, 5-fold Cross-Validation
- **UI/UX**: Tema kesehatan hijau-putih (Inter font), professional health portal look
- **Fitur tambahan**:
  - Hero landing page dengan statistik model + cara kerja + CTA
  - Form prediksi 1-kolom: hasil prediksi muncul di bawah form (bukan samping)
  - Tombol **"Isi dengan Data Contoh"** di pojok kanan atas card — memilih acak dari 6 profil pasien realistis
  - Tooltip ikon `ⓘ` pada kolom Nyeri Dada, EKG Istirahat, dan Kemiringan ST — penjelasan muncul saat hover tanpa memenuhi layar
  - Confidence score + probabilitas dua kelas + feature importance chart (top-10)
  - **Autentikasi**: daftar akun & masuk (SQLite + bcrypt hash via werkzeug)
  - **Riwayat prediksi**: disimpan otomatis ke database jika pengguna sudah login; ditampilkan sebagai accordion card yang bisa diklik
  - Navbar responsif: link Riwayat hanya muncul saat sudah login; dropdown profil menggunakan JS click-toggle (tidak hilang saat kursor berpindah)
  - Halaman Tentang: konteks medis, pipeline model, FAQ, disclaimer
- **Halaman**: Beranda, Prediksi, Masuk, Daftar, Riwayat, Tentang

### 3. Regresi — Gradient Boosting + House Price
- **Dataset**: Boston Housing (195 baris inline)
- **Model**: Gradient Boosting Regressor
- **Evaluasi**: R² Score, Mean Absolute Error (MAE)
- **Fitur tambahan**:
  - Halaman landing page dengan penjelasan interval kepercayaan
  - Interval kepercayaan prediksi harga (±MAE)
  - R² progress bar sebagai indikator kualitas model
  - Gauge bar posisi harga relatif terhadap rentang dataset
  - Input: RM, LSTAT, PTRATIO, CHAS, RAD

### 4. ARIMA — Time Series Whoosh Jakarta–Bandung
- **Dataset**: Data sintetis penumpang Kereta Cepat Whoosh (30 bulan, okt 2023–mar 2026)
- **Model**: ARIMA(1,1,1) dari statsmodels
- **Fitur tambahan**:
  - Chart interaktif (Chart.js): data historis + garis forecast
  - Pilih horizon prediksi (3–24 bulan)
  - Tabel rincian forecast per bulan dengan perubahan
  - Statistik ringkasan (rata-rata, max, min, tren)

---

## Setup Awal (lakukan sekali)

### 1. Buat virtual environment

```bash
cd contoh-flask
python -m venv venv
```

### 2. Aktifkan virtual environment

```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install semua dependensi

```bash
pip install -r requirements.txt
```

> File `requirements.txt` sudah mencakup semua dependensi yang dibutuhkan (Flask, scikit-learn, pandas, numpy, joblib, statsmodels, werkzeug, dll.).

---

## Cara Menjalankan

Setiap contoh dijalankan dari folder masing-masing.
**Pastikan virtual environment sudah aktif sebelum menjalankan.**

### Versi Sederhana (port 5000)
```bash
cd contoh-flask
python train_simple.py   # jalankan sekali
python app.py
```
Buka: http://127.0.0.1:5000

---

### Klasifikasi — HeartCheck (port 5001)
```bash
cd .\klasifikasi\
python train.py          # jalankan sekali (butuh file heart_vailure_dataset.csv)
python app.py
```
Buka: http://127.0.0.1:5001

| Route | Halaman |
|-------|---------|
| `/` | Beranda (Hero Landing Page) |
| `/prediksi
` | Form prediksi 11 fitur klinis |
| `/daftar` | Pendaftaran akun baru |
| `/masuk` | Masuk akun |
| `/keluar` | Keluar dari sesi |
| `/riwayat` | Riwayat prediksi (login diperlukan) |
| `/tentang` | Tentang aplikasi & model |

> Database SQLite (`heartcheck.db`) dibuat otomatis saat pertama kali `app.py` dijalankan.

---

### Regresi — House Price (port 5002)
```bash
cd .\regresi\
python train.py          # jalankan sekali
python app.py
```
Buka: http://127.0.0.1:5002
- `/`          → Landing page
- `/prediksi`  → Form estimasi harga properti

---

### ARIMA — Whoosh Passenger Forecast (port 5003)
```bash
cd .\arima\
python train.py          # jalankan sekali
python app.py
```
Buka: http://127.0.0.1:5003
- `/`          → Dashboard forecast penumpang Whoosh Jakarta–Bandung

---

## Konsep Machine Learning yang Didemonstrasikan

| Konsep | Aplikasi |
|--------|----------|
| Decision Tree | root/app.py |
| Random Forest + Pipeline | klasifikasi/ |
| Confidence Score (predict_proba) | semua klasifikasi |
| Feature Importance | klasifikasi/ |
| StandardScaler + OneHotEncoder | klasifikasi/ |
| Cross-Validation | klasifikasi/train.py |
| AUC-ROC | klasifikasi/ |
| Gradient Boosting Regressor | regresi/ |
| R² Score & MAE | regresi/ |
| Prediction Interval (±MAE) | regresi/ |
| ARIMA Time Series | arima/ |
| Chart.js Visualisasi | arima/ |
| Autentikasi + Session (Flask) | klasifikasi/ |
| SQLite + ORM sederhana | klasifikasi/database.py |
| Tooltip CSS + JS accordion UI | klasifikasi/templates/ |
| data-* attribute + JS style injection | klasifikasi/prediksi.html, riwayat.html |

---

## Catatan

- Setiap folder memiliki `model.joblib` sendiri — model tidak saling berbagi.
- `debug=True` hanya untuk development. Nonaktifkan di production.
- Port berbeda (5000–5003) agar semua bisa dijalankan bersamaan untuk perbandingan.
- Data klasifikasi memerlukan file CSV `heart_vailure_dataset.csv` di folder `klasifikasi/`.
- Data regresi dan ARIMA sudah disertakan langsung di dalam script (inline/sintetis).
