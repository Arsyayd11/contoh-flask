# klasifikasi/app.py
# ============================================================
# Aplikasi web HeartCheck — Deteksi Dini Penyakit Jantung
#
# Dataset  : Heart Failure Prediction (918 observasi, 11 fitur klinis)
# Model    : Random Forest Classifier dengan Pipeline preprocessing
#
# Daftar Route:
#   GET  /              → halaman landing (beranda)
#   GET  /prediksi      → form prediksi (kosong)
#   POST /prediksi      → proses prediksi, simpan riwayat jika login
#   GET  /register      → form pendaftaran akun
#   POST /register      → proses pendaftaran
#   GET  /login         → form masuk akun
#   POST /login         → proses masuk akun
#   GET  /logout        → keluar (hapus sesi)
#   GET  /riwayat       → riwayat prediksi (harus login)
#   GET  /tentang       → halaman tentang aplikasi & model
#
# Cara menjalankan:
#   1. python train.py   (sekali, untuk menghasilkan model.joblib)
#   2. python app.py
#   3. Buka http://127.0.0.1:5001
# ============================================================

import os

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash
)
from functools import wraps
import joblib
import pandas as pd

# Impor fungsi-fungsi database yang telah kita buat di database.py
from database import init_db, register_user, login_user, simpan_prediksi, ambil_riwayat

# ── Inisialisasi Flask ────────────────────────────────────────────────────────
app = Flask(__name__)

# secret_key WAJIB ada agar Flask bisa mengenkripsi data sesi (cookie).
# Di production, gunakan nilai acak yang panjang dan simpan di environment variable.
app.secret_key = "heartcheck_secret_key_2024_ganti_di_production"

# ── Inisialisasi database SQLite ──────────────────────────────────────────────
# Fungsi init_db() membuat file heartcheck.db dan tabel-tabelnya jika belum ada.
# Dipanggil sekali saat aplikasi pertama kali dijalankan.
init_db()

# ── Muat model dan metadata (sekali saat server start) ───────────────────────
# Model disimpan sebagai dictionary oleh train.py.
# Memuat di luar route agar tidak dimuat ulang setiap request (lebih efisien).
data_model = joblib.load("model.joblib")
pipeline   = data_model["pipeline"]          # Pipeline = preprocessor + RandomForest
ACCURACY   = data_model["accuracy"]          # akurasi pada data uji (float 0–1)
AUC        = data_model["auc"]               # AUC-ROC score pada data uji
FI_NAMES   = data_model["feature_names"]     # nama semua fitur setelah encoding
FI_VALUES  = data_model["importances"]       # bobot feature importance dari model


# ── Helper: decorator login_required ─────────────────────────────────────────
# Decorator ini membungkus route agar hanya bisa diakses oleh pengguna yang sudah login.
# Jika belum login → redirect ke halaman login dengan pesan peringatan.
# Pola ini mirip dengan @login_required di Flask-Login, tapi dibuat manual.
def login_required(f):
    @wraps(f)       # @wraps mempertahankan nama dan docstring fungsi asli
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Silakan masuk terlebih dahulu untuk mengakses halaman ini.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ── Helper: ambil data user dari session ─────────────────────────────────────
# Fungsi kecil untuk menyederhanakan pengambilan info user yang sedang login.
def current_user():
    return {
        "id"   : session.get("user_id"),
        "nama" : session.get("user_nama"),
        "email": session.get("user_email"),
    } if "user_id" in session else None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 1: Beranda / Landing Page
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def beranda():
    """
    Halaman utama (landing page) yang menampilkan:
    - Hero section dengan CTA ke halaman prediksi
    - Statistik performa model (akurasi, AUC)
    - Penjelasan cara kerja aplikasi
    """
    return render_template(
        "landing.html",
        accuracy=round(ACCURACY * 100, 1),   # contoh: 88.0 (persen)
        auc=AUC,                              # contoh: 0.9231
        user=current_user(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 2: Halaman Prediksi
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    """
    GET  → tampilkan form input 11 fitur klinis (kosong / siap diisi)
    POST → baca input dari form, jalankan prediksi, tampilkan hasil

    Jika pengguna sudah login dan prediksi berhasil:
    → hasil prediksi otomatis disimpan ke database (tabel riwayat_prediksi)
    """
    result = None
    error  = None

    if request.method == "POST":
        try:
            # ── Tahap 1: Baca semua input dari form HTML ──────────────────────
            # Nama key harus sama persis dengan atribut name="..." di template.
            # int() / float() akan memicu ValueError jika input bukan angka.
            age             = float(request.form["age"])
            sex             = request.form["sex"]               # "M" atau "F"
            chest_pain      = request.form["chest_pain"]        # TA / ATA / NAP / ASY
            resting_bp      = float(request.form["resting_bp"])
            cholesterol     = float(request.form["cholesterol"])
            fasting_bs      = int(request.form["fasting_bs"])   # 0 atau 1
            resting_ecg     = request.form["resting_ecg"]       # Normal / ST / LVH
            max_hr          = float(request.form["max_hr"])
            exercise_angina = request.form["exercise_angina"]   # "Y" atau "N"
            oldpeak         = float(request.form["oldpeak"])
            st_slope        = request.form["st_slope"]          # Up / Flat / Down

            # ── Tahap 2: Susun DataFrame input untuk pipeline ─────────────────
            # Pipeline mengharapkan DataFrame dengan nama kolom yang sama seperti
            # saat training. Urutan: fitur_numerik lebih dahulu, lalu kategorikal.
            # (lihat klasifikasi/train.py untuk urutan yang benar)
            input_df = pd.DataFrame([{
                "Age"            : age,
                "RestingBP"      : resting_bp,
                "Cholesterol"    : cholesterol,
                "FastingBS"      : fasting_bs,
                "MaxHR"          : max_hr,
                "Oldpeak"        : oldpeak,
                "Sex"            : sex,
                "ChestPainType"  : chest_pain,
                "RestingECG"     : resting_ecg,
                "ExerciseAngina" : exercise_angina,
                "ST_Slope"       : st_slope,
            }])

            # ── Tahap 3: Prediksi menggunakan Pipeline ────────────────────────
            # pipeline.predict()       → [0] atau [1]  (kelas prediksi)
            # pipeline.predict_proba() → [[p_normal, p_disease]]  (probabilitas)
            # Pipeline secara otomatis menerapkan preprocessing sebelum prediksi.
            pred_class = pipeline.predict(input_df)[0]
            proba      = pipeline.predict_proba(input_df)[0]
            confidence = round(float(max(proba)) * 100, 1)   # keyakinan tertinggi

            # ── Tahap 4: Hitung feature importance untuk visualisasi ──────────
            # Nilai importance sudah dihitung saat training dan disimpan di model.joblib.
            # Kita hanya perlu mengurutkan dan memformat ulang untuk ditampilkan.
            fi_list = sorted(
                zip(FI_NAMES, [round(v * 100, 2) for v in FI_VALUES]),
                key=lambda x: x[1], reverse=True
            )[:10]   # hanya tampilkan 10 fitur paling berpengaruh

            # ── Tahap 5: Kemas hasil ke dict untuk template ───────────────────
            hasil_label = "Risiko Tinggi" if pred_class == 1 else "Risiko Rendah"
            result = {
                "label"          : hasil_label,
                "disease"        : bool(pred_class),
                "confidence"     : confidence,
                "prob_disease"   : round(float(proba[1]) * 100, 1),
                "prob_normal"    : round(float(proba[0]) * 100, 1),
                "fi"             : fi_list,
                # Ringkasan input dalam format yang mudah dibaca
                "age"            : int(age),
                "sex_label"      : "Laki-laki" if sex == "M" else "Perempuan",
                "chest_pain"     : chest_pain,
                "resting_bp"     : resting_bp,
                "cholesterol"    : cholesterol,
                "fasting_bs_lbl" : "Ya (> 120 mg/dl)" if fasting_bs == 1 else "Tidak",
                "resting_ecg"    : resting_ecg,
                "max_hr"         : int(max_hr),
                "exercise_angina": "Ya" if exercise_angina == "Y" else "Tidak",
                "oldpeak"        : oldpeak,
                "st_slope"       : st_slope,
            }

            # ── Tahap 6: Simpan riwayat jika pengguna sudah login ────────────
            # Pengguna tamu (belum login) tetap bisa prediksi, tapi hasil tidak disimpan.
            # Pengguna yang sudah login → hasil tersimpan otomatis di database.
            if "user_id" in session:
                input_raw = {
                    "age": int(age), "sex": sex, "chest_pain": chest_pain,
                    "resting_bp": resting_bp, "cholesterol": cholesterol,
                    "fasting_bs": fasting_bs, "resting_ecg": resting_ecg,
                    "max_hr": int(max_hr), "exercise_angina": exercise_angina,
                    "oldpeak": oldpeak, "st_slope": st_slope,
                }
                simpan_prediksi(
                    session["user_id"], input_raw,
                    hasil_label, result["prob_disease"], confidence
                )

        except ValueError as e:
            error = f"Input tidak valid. Pastikan semua kolom angka diisi dengan benar. ({e})"

    return render_template(
        "prediksi.html",
        result=result,
        error=error,
        accuracy=round(ACCURACY * 100, 1),
        auc=AUC,
        user=current_user(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 3: Daftar Akun Baru
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/daftar", methods=["GET", "POST"])
def register():
    """
    GET  → tampilkan form pendaftaran
    POST → validasi input → simpan ke database → redirect ke login
    """
    # Jika sudah login, langsung ke beranda
    if "user_id" in session:
        return redirect(url_for("beranda"))

    if request.method == "POST":
        nama     = request.form.get("nama", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        konfirm  = request.form.get("konfirm_password", "")

        # ── Validasi server-side ──────────────────────────────────────────────
        if not nama or not email or not password:
            flash("Semua kolom wajib diisi.", "danger")
        elif password != konfirm:
            flash("Konfirmasi password tidak cocok.", "danger")
        elif len(password) < 8:
            flash("Password minimal 8 karakter.", "danger")
        else:
            # Coba simpan ke database; False berarti email sudah terdaftar
            berhasil = register_user(nama, email, password)
            if berhasil:
                flash("Akun berhasil dibuat! Silakan masuk.", "success")
                return redirect(url_for("login"))
            else:
                flash("Email sudah terdaftar. Gunakan email lain.", "danger")

    return render_template("register.html", user=current_user())


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 4: Masuk Akun
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/masuk", methods=["GET", "POST"])
def login():
    """
    GET  → tampilkan form login
    POST → verifikasi email + password → simpan sesi → redirect ke beranda
    """
    if "user_id" in session:
        return redirect(url_for("beranda"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # login_user() mengembalikan dict user jika valid, None jika gagal
        user = login_user(email, password)
        if user:
            # Simpan data user ke sesi Flask (disimpan di cookie terenkripsi)
            session["user_id"]    = user["id"]
            session["user_nama"]  = user["nama"]
            session["user_email"] = user["email"]
            flash(f"Selamat datang kembali, {user['nama']}!", "success")
            return redirect(url_for("beranda"))
        else:
            flash("Email atau password salah.", "danger")

    return render_template("login.html", user=current_user())


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 5: Keluar
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/keluar")
def logout():
    """
    Hapus semua data sesi → pengguna kembali menjadi tamu.
    session.clear() menghapus semua key dari sesi Flask.
    """
    session.clear()
    flash("Anda telah keluar dari akun.", "info")
    return redirect(url_for("beranda"))


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 6: Riwayat Prediksi (hanya untuk pengguna yang login)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/riwayat")
@login_required   # decorator: redirect ke login jika belum masuk
def riwayat():
    """
    Tampilkan semua riwayat prediksi milik pengguna yang sedang login.
    Data diambil dari tabel riwayat_prediksi berdasarkan user_id di sesi.
    """
    data_riwayat = ambil_riwayat(session["user_id"])
    return render_template(
        "riwayat.html",
        riwayat=data_riwayat,
        user=current_user(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 7: Tentang Aplikasi
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/tentang")
def tentang():
    """
    Halaman informasi tentang aplikasi HeartCheck:
    konteks medis, dataset, algoritma, dan cara penggunaan.
    """
    return render_template(
        "tentang.html",
        accuracy=round(ACCURACY * 100, 1),
        auc=AUC,
        user=current_user(),
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True: reload otomatis saat kode berubah + tampilkan error detail.
    # Port 5001 agar tidak bentrok dengan aplikasi lain (root=5000, dll).
    # JANGAN gunakan debug=True di production server
    port = int(os.environ.get("PORT", 5001))

    app.run(
        host="0.0.0.0",   # WAJIB agar bisa diakses dari luar
        port=port,
        debug=False       # WAJIB dimatikan di server
    )