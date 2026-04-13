# klasifikasi/train.py
# ============================================================
# Dataset  : Heart Failure Prediction Dataset
#            Sumber: Fedesoriano (2021), Kaggle
#            File  : heart_vailure_dataset.csv (918 observasi, 11 fitur)
#
# Konteks  :
#   Penyakit kardiovaskular (CVD) adalah penyebab kematian nomor 1 di dunia,
#   menewaskan sekitar 17.9 juta jiwa per tahun (31% dari total kematian global).
#   Model machine learning dapat membantu deteksi dini risiko penyakit jantung.
#
# Fitur    :
#   Numerik   : Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
#   Kategorikal: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
#
# Target   : HeartDisease (1 = ada penyakit, 0 = normal)
#
# Model    : Random Forest Classifier dengan Pipeline preprocessing
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib

# ── 1. Muat dataset dari file CSV ────────────────────────────────────────────
# Menggunakan path absolut agar bisa dijalankan dari direktori mana pun
df = pd.read_csv(
    r"C:\Users\arsya\Documents\Semester7\Project\contoh-flask\klasifikasi\heart_vailure_dataset.csv"
)

print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"Distribusi target:\n{df['HeartDisease'].value_counts()}\n")

# ── 2. Definisi fitur dan target ─────────────────────────────────────────────
# Fitur numerik → akan di-scaling dengan StandardScaler
# Fitur kategorikal → akan di-encode dengan OneHotEncoder
fitur_numerik      = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
fitur_kategorikal  = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
target             = "HeartDisease"

X = df[fitur_numerik + fitur_kategorikal]   # semua 11 fitur
y = df[target]                              # label (0 atau 1)

# ── 3. Train-test split (stratified) ─────────────────────────────────────────
# stratify=y → memastikan proporsi kelas positif & negatif sama di train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data training: {X_train.shape[0]} baris")
print(f"Data pengujian: {X_test.shape[0]} baris\n")

# ── 4. Preprocessing pipeline ────────────────────────────────────────────────
# ColumnTransformer menerapkan transformasi berbeda per kolom:
#   - StandardScaler  : normalisasi fitur numerik (mean=0, std=1)
#   - OneHotEncoder   : ubah kategori teks menjadi kolom biner (dummy variable)
#     handle_unknown="ignore" → nilai baru saat prediksi tidak akan error
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                                               fitur_numerik),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),    fitur_kategorikal),
])

# Pipeline menggabungkan preprocessing + model secara berurutan.
# Keuntungan: preprocessing otomatis diterapkan saat predict() dipanggil.
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  RandomForestClassifier(
        n_estimators=200,       # jumlah pohon keputusan dalam forest
        max_depth=10,           # kedalaman maksimum setiap pohon
        min_samples_leaf=3,     # min sampel per daun (mencegah overfitting)
        class_weight="balanced", # kompensasi imbalance jika ada
        random_state=42
    )),
])

# ── 5. Training ───────────────────────────────────────────────────────────────
print("Melatih model...")
pipeline.fit(X_train, y_train)

# ── 6. Evaluasi pada data uji ─────────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probabilitas kelas positif (HeartDisease=1)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*50}")
print(f"Akurasi   : {acc:.2%}")
print(f"AUC-ROC   : {auc:.4f}   (1.0 = sempurna, 0.5 = acak)")
print(f"{'='*50}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Heart Disease"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 7. Cross-validation (evaluasi lebih robust) ───────────────────────────────
# K-Fold CV membagi seluruh data menjadi K lipatan, melatih K kali,
# menghasilkan estimasi performa yang lebih stabil dari satu split saja.
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_sc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
print(f"\nCross-Validation AUC-ROC (5-fold): {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

# ── 8. Ambil nama fitur setelah encoding ──────────────────────────────────────
# OneHotEncoder menghasilkan kolom baru misal "Sex_M", "Sex_F", dll.
# Kita butuh nama-nama ini untuk menampilkan feature importance yang informatif.
clf           = pipeline.named_steps["clf"]
prep          = pipeline.named_steps["prep"]
cat_names     = prep.named_transformers_["cat"].get_feature_names_out(fitur_kategorikal)
all_feat_names = list(fitur_numerik) + list(cat_names)

# ── 9. Simpan model + metadata ke disk ───────────────────────────────────────
# Menyimpan sebagai dictionary agar app.py bisa membaca metadata tanpa melatih ulang
joblib.dump({
    "pipeline"          : pipeline,         # seluruh pipeline (prep + model)
    "accuracy"          : round(acc, 4),    # akurasi pada data uji
    "auc"               : round(auc, 4),    # AUC-ROC pada data uji
    "fitur_numerik"     : fitur_numerik,
    "fitur_kategorikal" : fitur_kategorikal,
    "feature_names"     : all_feat_names,   # nama fitur setelah encoding
    "importances"       : clf.feature_importances_.tolist(),  # bobot tiap fitur
}, "model.joblib")

print("\nklasifikasi/model.joblib tersimpan.")
print("Jalankan 'python app.py' untuk memulai server prediksi.")
