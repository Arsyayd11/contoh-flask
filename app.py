# app.py — versi sederhana (untuk penjelasan dasar Flask)
# Dataset: Titanic | Model: Decision Tree Classifier
#
# ALUR KERJA APLIKASI:
#   1. Jalankan train_simple.py SEKALI untuk melatih model & menyimpan model.joblib
#   2. Jalankan app.py → Flask muat model dari disk
#   3. Setiap kali user submit form → Flask baca input → prediksi → kirim ke HTML
#
# Cara menjalankan:
#   1. python train_simple.py   (cukup sekali, sampai model.joblib ada)
#   2. python app.py
#   3. Buka browser → http://127.0.0.1:5000

from flask import Flask, render_template, request
import joblib
import numpy as np

# ── Inisialisasi aplikasi Flask ──────────────────────────────────────────────
# Flask(__name__) memberi tahu Flask di mana mencari folder templates/ & static/
app = Flask(__name__)

# ── Muat model satu kali saat server dinyalakan ──────────────────────────────
# Ditulis di luar fungsi route agar tidak dimuat ulang setiap request.
# model.joblib berisi dict {"model": ..., "accuracy": ...} yang disimpan train_simple.py
data_model = joblib.load("model.joblib")
model      = data_model["model"]      # objek DecisionTreeClassifier
ACCURACY   = data_model["accuracy"]   # akurasi pada data uji (misal: 0.8250)


# ── Route utama ──────────────────────────────────────────────────────────────
# @app.route mendaftarkan fungsi sebagai "handler" untuk URL tertentu.
# methods=["GET", "POST"] berarti route ini menerima dua jenis request HTTP:
#   GET  → saat user membuka URL langsung (pertama kali buka halaman)
#   POST → saat user submit form (menekan tombol Prediksi)
@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  → tampilkan form kosong, belum ada prediksi
    POST → baca input form, lakukan prediksi, kembalikan hasil ke template
    """
    prediction = None   # hasil label prediksi ("Selamat" / "Tidak Selamat")
    confidence = None   # tingkat keyakinan model dalam persen
    error      = None   # pesan error jika input tidak valid

    if request.method == "POST":
        try:
            # ── Tahap 1: Baca input dari form HTML ───────────────────────────
            # request.form adalah dictionary berisi data dari <input> dan <select>
            # Nama key harus sama dengan atribut name="..." di HTML
            pclass = int(request.form["pclass"])     # dropdown → integer (1, 2, atau 3)
            sex    = request.form["sex"]             # dropdown → string ("male"/"female")
            age    = float(request.form["age"])      # input angka → float
            fare   = float(request.form["fare"])     # input angka → float

            # ── Tahap 2: Encode fitur kategorikal ────────────────────────────
            # Machine learning membutuhkan angka, bukan teks.
            # Encoding HARUS identik dengan yang digunakan saat training!
            # (lihat train_simple.py baris df["Sex"] = df["Sex"].map(...))
            sex_encoded = 0 if sex == "male" else 1  # male→0, female→1

            # ── Tahap 3: Susun array input ────────────────────────────────────
            # np.array([[...]]) → membentuk array 2D dengan shape (1, 4)
            # Urutan kolom WAJIB sama dengan urutan X_train saat training:
            # [Pclass, Sex, Age, Fare]
            features = np.array([[pclass, sex_encoded, age, fare]])

            # ── Tahap 4: Prediksi ──────────────────────────────────────────────
            # model.predict()       → mengembalikan label kelas [0] atau [1]
            # model.predict_proba() → mengembalikan probabilitas tiap kelas
            #   format: [[prob_kelas_0, prob_kelas_1]]
            #   contoh: [[0.35, 0.65]] → 65% kemungkinan selamat
            hasil      = model.predict(features)[0]
            proba      = model.predict_proba(features)[0]  # array [prob_0, prob_1]
            confidence = round(float(max(proba)) * 100, 1)  # ambil probabilitas tertinggi

            # Ubah angka kelas menjadi label yang mudah dipahami
            prediction = "Selamat" if hasil == 1 else "Tidak Selamat"

        except ValueError:
            # ValueError terjadi jika input bukan angka (misal: user ketik huruf)
            error = "Input tidak valid. Pastikan Age dan Fare diisi dengan angka."

    # ── Tahap 5: Kirim data ke template HTML ──────────────────────────────────
    # render_template() membaca file templates/index.html dan mengisi variabel Jinja2
    # Semua variabel yang dikirim bisa diakses di HTML dengan {{ nama_variabel }}
    return render_template(
        "index.html",
        prediction=prediction,    # "Selamat", "Tidak Selamat", atau None
        confidence=confidence,    # angka persen, misal: 78.5
        error=error,              # pesan error atau None
        accuracy=round(ACCURACY * 100, 1),  # akurasi dalam persen, misal: 82.5
    )


# ── Entry point ──────────────────────────────────────────────────────────────
# Blok ini hanya berjalan saat file ini dieksekusi langsung (bukan di-import)
if __name__ == "__main__":
    # debug=True: server auto-restart saat kode berubah + tampilkan error detail di browser
    # JANGAN gunakan debug=True di production server!
    app.run(debug=True)
