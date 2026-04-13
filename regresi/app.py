# regresi/app.py
# ============================================================
# Dataset  : Boston Housing (subset 195 observasi)
# Model    : Gradient Boosting Regressor
# Target   : Prediksi harga rumah (MEDV, dalam $1000)
# Fitur    : RM, LSTAT, PTRATIO, CHAS, RAD
#
# Route    :
#   GET  /           → landing page (penjelasan + performa model)
#   GET  /prediksi   → form estimasi harga
#   POST /prediksi   → proses prediksi + interval kepercayaan
#
# Cara menjalankan:
#   1. python train.py   (sekali, untuk menghasilkan model.joblib)
#   2. python app.py
#   3. Buka http://127.0.0.1:5002
# ============================================================

from flask import Flask, render_template, request
import joblib
import numpy as np

# ── Inisialisasi Flask ────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Muat model dan metadata ────────────────────────────────────────────────────
# train.py menyimpan dict {"model": ..., "r2": ..., "mae": ..., "mae_usd": ...}
data_model = joblib.load("model.joblib")
model      = data_model["model"]      # GradientBoostingRegressor
R2         = data_model["r2"]         # koefisien determinasi (misal: 0.8823)
MAE        = data_model["mae"]        # dalam ribuan USD (misal: 2.341)
MAE_USD    = data_model["mae_usd"]    # dalam USD (misal: 2341)

# Rentang referensi dari data training (untuk menghitung posisi persentil gauge)
MEDV_MIN, MEDV_MAX = 5.0, 50.0


# ── Route 1: Landing Page ──────────────────────────────────────────────────────
@app.route("/")
def landing():
    """Halaman awal: konteks, performa model, penjelasan fitur."""
    return render_template(
        "landing.html",
        r2=R2,
        mae_usd=MAE_USD,
    )


# ── Route 2: Form Prediksi ─────────────────────────────────────────────────────
@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    """
    GET  → tampilkan form kosong
    POST → prediksi harga + interval kepercayaan ±MAE
    """
    result = None
    error  = None

    if request.method == "POST":
        try:
            # ── Baca input dari form ─────────────────────────────────────────
            rm      = float(request.form["rm"])       # rata-rata jumlah kamar
            lstat   = float(request.form["lstat"])    # % populasi kelas bawah
            ptratio = float(request.form["ptratio"])  # rasio murid/guru
            chas    = int(request.form["chas"])        # 0 atau 1 (pinggir sungai)
            rad     = int(request.form["rad"])         # 1–8 (aksesibilitas jalan raya)

            # ── Prediksi ──────────────────────────────────────────────────────
            features = np.array([[rm, lstat, ptratio, chas, rad]])
            prediksi_val = float(model.predict(features)[0])

            # ── Confidence interval ±MAE ────────────────────────────────────
            # Untuk regresi, kita tidak punya probabilitas seperti klasifikasi.
            # Sebagai gantinya, kita tunjukkan interval kepercayaan berbasis MAE:
            #   Harga prediksi ± MAE → rentang yang 'masuk akal' untuk nilai sebenarnya.
            batas_bawah = max(prediksi_val - MAE, 0)   # tidak boleh negatif
            batas_atas  = prediksi_val + MAE

            # ── Posisi persentil untuk gauge bar ────────────────────────────
            pct = min(max((prediksi_val - MEDV_MIN) / (MEDV_MAX - MEDV_MIN) * 100, 0), 100)

            # ── R² sebagai indikator kualitas model ─────────────────────────
            # R² = 1.0 → model sempurna; R² = 0.0 → model seperti prediksi rata-rata saja
            r2_pct = round(R2 * 100, 1)   # dalam persen (lebih intuitif)

            result = {
                "harga"        : round(prediksi_val * 1000),       # dalam USD
                "harga_k"      : round(prediksi_val, 1),           # dalam ribuan USD
                "bawah"        : round(batas_bawah * 1000),        # interval bawah USD
                "atas"         : round(batas_atas * 1000),         # interval atas USD
                "bawah_k"      : round(batas_bawah, 1),
                "atas_k"       : round(batas_atas, 1),
                "pct"          : round(pct, 1),
                "r2_pct"       : r2_pct,
                "mae_usd"      : MAE_USD,
                # Ringkasan input
                "rm"           : rm,
                "lstat"        : lstat,
                "ptratio"      : ptratio,
                "chas"         : "Ya" if chas == 1 else "Tidak",
                "rad"          : rad,
            }

        except ValueError:
            error = "Input tidak valid. Pastikan semua kolom angka diisi dengan benar."

    return render_template(
        "index.html",
        result=result,
        error=error,
        r2=R2,
        r2_pct=round(R2 * 100, 1),
        mae_usd=MAE_USD,
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Port 5002 agar tidak bertabrakan dengan app lain
    app.run(debug=True, port=5002)
