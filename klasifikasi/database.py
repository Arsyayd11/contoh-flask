# klasifikasi/database.py
# ============================================================
# Modul pengelola database SQLite untuk aplikasi HeartCheck.
#
# Tabel:
#   users              → data akun pengguna (nama, email, password)
#   riwayat_prediksi   → histori prediksi tiap pengguna
#
# Fungsi yang tersedia:
#   init_db()          → inisialisasi / buat tabel jika belum ada
#   get_db()           → buka koneksi ke database
#   register_user()    → daftarkan akun baru
#   login_user()       → verifikasi kredensial login
#   simpan_prediksi()  → simpan satu record hasil prediksi
#   ambil_riwayat()    → ambil semua riwayat prediksi milik user
# ============================================================

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Path absolut ke file database, relatif terhadap lokasi file ini.
# Disimpan di folder klasifikasi/ sebagai heartcheck.db.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heartcheck.db")


def get_db():
    """
    Buka koneksi ke database SQLite.
    row_factory = sqlite3.Row → baris bisa diakses seperti dict (row["kolom"]).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Inisialisasi database: buat tabel-tabel yang dibutuhkan jika belum ada.
    Dipanggil sekali saat aplikasi pertama kali dijalankan.
    """
    conn = get_db()
    cur  = conn.cursor()

    # ── Tabel pengguna ────────────────────────────────────────────────────────
    # Menyimpan data akun: nama, email (unik), password (di-hash).
    # TIDAK menyimpan password plaintext — selalu gunakan hash!
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER  PRIMARY KEY AUTOINCREMENT,
            nama          TEXT     NOT NULL,
            email         TEXT     UNIQUE NOT NULL,
            password_hash TEXT     NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Tabel riwayat prediksi ────────────────────────────────────────────────
    # Setiap baris mewakili satu sesi prediksi yang dilakukan pengguna.
    # Menyimpan semua input fitur klinis beserta hasil prediksi model.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_prediksi (
            id              INTEGER  PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER  NOT NULL,
            age             INTEGER,
            sex             TEXT,
            chest_pain      TEXT,
            resting_bp      REAL,
            cholesterol     REAL,
            fasting_bs      INTEGER,
            resting_ecg     TEXT,
            max_hr          REAL,
            exercise_angina TEXT,
            oldpeak         REAL,
            st_slope        TEXT,
            hasil           TEXT,
            prob_disease    REAL,
            confidence      REAL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def register_user(nama, email, password):
    """
    Daftarkan pengguna baru ke database.

    Password di-hash dengan bcrypt (via werkzeug) sebelum disimpan —
    sehingga bahkan admin database tidak bisa melihat password asli.

    Return:
        True  → pendaftaran berhasil
        False → email sudah terdaftar (IntegrityError karena UNIQUE constraint)
    """
    conn = get_db()
    try:
        password_hash = generate_password_hash(password)  # hash password
        conn.execute(
            "INSERT INTO users (nama, email, password_hash) VALUES (?, ?, ?)",
            (nama, email, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Email sudah ada di database → tidak boleh duplikat
        return False
    finally:
        conn.close()


def login_user(email, password):
    """
    Verifikasi kredensial login pengguna.

    Proses:
      1. Cari user berdasarkan email
      2. Bandingkan password input dengan hash tersimpan menggunakan check_password_hash()
      3. Jika cocok → kembalikan data user sebagai dict

    Return:
        dict  → user data jika login valid
        None  → email tidak ditemukan atau password salah
    """
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()

    # check_password_hash() membandingkan plaintext dengan hash secara aman
    if user and check_password_hash(user["password_hash"], password):
        return dict(user)   # konversi sqlite3.Row ke dict biasa
    return None


def simpan_prediksi(user_id, input_data, hasil, prob_disease, confidence):
    """
    Simpan satu record hasil prediksi ke tabel riwayat_prediksi.

    Parameter:
        user_id     → ID pengguna yang melakukan prediksi
        input_data  → dict berisi 11 nilai fitur klinis
        hasil       → string "Risiko Tinggi" atau "Risiko Rendah"
        prob_disease→ probabilitas penyakit jantung (0.0–1.0 × 100)
        confidence  → confidence score tertinggi (persen)
    """
    conn = get_db()
    conn.execute("""
        INSERT INTO riwayat_prediksi
        (user_id, age, sex, chest_pain, resting_bp, cholesterol,
         fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak,
         st_slope, hasil, prob_disease, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        input_data["age"],
        input_data["sex"],
        input_data["chest_pain"],
        input_data["resting_bp"],
        input_data["cholesterol"],
        input_data["fasting_bs"],
        input_data["resting_ecg"],
        input_data["max_hr"],
        input_data["exercise_angina"],
        input_data["oldpeak"],
        input_data["st_slope"],
        hasil,
        prob_disease,
        confidence,
    ))
    conn.commit()
    conn.close()


def ambil_riwayat(user_id):
    """
    Ambil semua riwayat prediksi milik pengguna tertentu,
    diurutkan dari yang terbaru (ORDER BY created_at DESC).

    Return:
        list[dict] → daftar record prediksi
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM riwayat_prediksi WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
