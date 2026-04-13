# arima/train.py
# ============================================================
# Dataset  : Data pengguna Kereta Cepat Whoosh (Jakarta–Bandung)
#            Data sintetis deterministik untuk keperluan demonstrasi.
#
# Konteks  :
#   Kereta Cepat Whoosh (Waktu Hemat Operasi Optimal Sistem Hebat) adalah
#   kereta cepat pertama di Asia Tenggara yang menghubungkan Jakarta dan
#   Bandung. Beroperasi sejak Oktober 2023, Whoosh mampu menempuh jarak
#   ±142 km dalam waktu sekitar 45 menit dengan kecepatan hingga 350 km/jam.
#
# Tujuan   :
#   Melatih model ARIMA(1,1,1) untuk memperkirakan jumlah penumpang
#   Whoosh rute Jakarta–Bandung pada bulan-bulan mendatang.
#
# Model    : ARIMA(1,1,1) dari statsmodels
# ============================================================

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate data penumpang bulanan (sintetis, deterministik) ─────────────
# Data dimulai Oktober 2023 (bulan operasi perdana Whoosh) hingga Maret 2026.
# Karakteristik data:
#   - Tren naik bertahap: penumpang tumbuh seiring popularitas meningkat
#   - Musiman: lonjakan saat Lebaran (April/Mei), liburan sekolah (Juni-Juli),
#              dan akhir tahun (Desember-Januari)
#   - Noise acak terkontrol untuk mensimulasikan variasi alami

np.random.seed(42)

# Periode data: Oktober 2023 – Maret 2026 = 30 bulan
dates = pd.date_range(start="2023-10-01", periods=30, freq="ME")
n     = len(dates)
t     = np.arange(n)

# Komponen data:
#   base        : penumpang awal (bulan perdana)
#   tren        : pertumbuhan linear ~15.000 penumpang/bulan
#   musiman     : pola tahunan berperiode 12 bulan (amplitudo 60.000)
#   lonjakan    : event khusus Lebaran (bulan April ke-6, ke-18) dan liburan
#   noise       : variasi acak ±20.000 penumpang

base    = 180_000
tren    = 15_000 * t
musiman = 60_000 * np.sin(2 * np.pi * t / 12 - np.pi / 2)   # palung di awal, puncak di pertengahan
noise   = np.random.normal(0, 20_000, n)

# Tambahkan lonjakan khusus pada bulan-bulan tertentu
lonjakan = np.zeros(n)
bulan_lebaran   = [6, 18]   # indeks bulan ke-7 dan ke-19 = April 2024 & 2025
bulan_libur     = [8, 9, 20, 21]  # Juni–Juli 2024 & 2025
bulan_tahunbaru = [3, 15, 27]     # Januari 2024, 2025, 2026

for idx in bulan_lebaran:
    if idx < n:
        lonjakan[idx] += 80_000
for idx in bulan_libur:
    if idx < n:
        lonjakan[idx] += 50_000
for idx in bulan_tahunbaru:
    if idx < n:
        lonjakan[idx] += 40_000

# Gabungkan semua komponen dan bulatkan ke ribuan terdekat
penumpang = (base + tren + musiman + lonjakan + noise).round(-3)
penumpang = np.maximum(penumpang, 50_000)  # minimal 50.000 penumpang/bulan

# Buat Series pandas dengan indeks waktu
series = pd.Series(
    penumpang.astype(int),
    index=dates,
    name="penumpang_whoosh_jakarta_bandung"
)

print("Data penumpang Whoosh Jakarta–Bandung (30 bulan terakhir):")
print(series.to_string())
print(f"\nRata-rata: {series.mean():,.0f} penumpang/bulan")
print(f"Maksimum : {series.max():,.0f} penumpang/bulan")
print(f"Minimum  : {series.min():,.0f} penumpang/bulan")

# ── 2. Fit model ARIMA(1,1,1) ────────────────────────────────────────────────
# ARIMA(p, d, q):
#   p=1 → satu lag autoregressive (AR) — pengaruh bulan lalu
#   d=1 → diferensiasi satu kali untuk membuat stasioner
#   q=1 → satu lag moving average (MA) — koreksi error bulan lalu
print("\nMelatih model ARIMA(1,1,1)...")
model_fit = ARIMA(series, order=(1, 1, 1)).fit()
print(f"AIC: {model_fit.aic:.2f}  (semakin rendah = model lebih baik)")

# ── 3. Simpan model + series ke disk ─────────────────────────────────────────
# model_fit : objek ARIMAResultsWrapper untuk forecast
# series    : data historis untuk ditampilkan di chart
joblib.dump({"model": model_fit, "series": series}, "model.joblib")
print("\narima/model.joblib tersimpan.")
print("Jalankan 'python app.py' untuk memulai server prediksi Whoosh.")
