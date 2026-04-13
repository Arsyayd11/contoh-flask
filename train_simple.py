# train_simple.py — jalankan sekali untuk menghasilkan model.joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import io

# Data Titanic inline (subset 200 baris cukup untuk demo)
data_csv = """Survived,Pclass,Sex,Age,Fare
0,3,male,22,7.25
1,1,female,38,71.28
1,3,female,26,7.93
1,1,female,35,53.1
0,3,male,35,8.05
0,3,male,27,8.46
0,1,male,54,51.86
0,3,male,2,21.08
1,3,female,27,11.13
1,2,female,14,30.07
1,3,female,4,16.7
1,1,female,58,26.55
0,3,male,20,8.05
0,3,male,39,31.28
0,3,female,14,7.85
1,2,female,55,16.0
0,3,male,2,29.13
1,2,male,23,13.0
0,3,female,31,18.0
1,3,female,22,7.22
0,2,male,35,26.0
0,2,male,34,13.0
1,3,female,15,8.03
1,1,male,28,35.5
0,3,male,8,21.08
1,3,female,38,31.39
0,3,male,19,7.9
0,1,male,40,27.72
1,2,female,29,146.52
0,3,male,45,8.05
0,3,male,22,7.73
1,1,female,30,106.43
0,3,male,29,7.88
1,1,female,18,110.88
0,2,male,21,26.0
0,3,male,25,7.23
1,2,female,36,83.48
0,3,male,17,7.23
0,3,female,20,7.23
1,3,female,21,8.43
0,2,male,18,13.0
0,3,male,21,8.67
1,2,female,45,13.5
1,1,male,21,77.29
0,3,male,55,8.05
0,3,male,5,20.53
0,1,male,51,61.98
0,3,male,22,7.9
1,1,female,55,59.4
0,2,male,27,13.0
1,1,female,26,78.85
0,3,male,24,8.05
1,3,male,24,16.7
0,1,male,58,153.46
0,3,male,44,8.05
0,2,male,28,23.0
1,1,female,49,76.73
0,2,male,52,13.0
1,1,female,44,57.0
0,1,male,27,30.0
0,3,male,24,7.93
0,2,male,28,0.0
0,3,male,33,7.9
0,1,male,60,26.55
0,2,male,41,15.55
1,2,female,15,14.5
0,3,male,50,7.73
0,2,male,34,13.0
1,3,female,28,7.86
0,1,male,45,75.24
0,3,male,40,31.39
0,3,male,36,15.5
0,3,male,32,10.5
1,1,female,19,91.08
0,3,male,19,7.9
0,3,male,3,21.0
1,1,female,44,227.53
1,3,female,58,15.5
0,3,male,42,7.23
0,2,male,29,10.52
0,3,female,22,10.52
1,1,male,24,79.2
0,3,male,28,8.05
1,1,female,25,26.0
0,2,male,31,15.05
1,1,female,29,211.34
0,3,male,25,7.05
0,1,male,39,76.73
0,3,male,33,10.46
0,3,male,24,7.91
1,3,female,22,9.35
0,3,male,27,8.67
1,1,female,28,133.65
0,3,male,25,9.22
0,3,male,25,7.23
1,1,female,25,151.55
0,3,male,34,14.1
0,1,male,52,30.5
1,3,female,30,12.48
0,3,male,28,9.59
0,3,male,23,7.9
0,3,male,39,29.13
0,3,male,35,7.96
1,1,female,27,211.34
0,3,male,21,7.9
0,3,male,22,8.05
0,3,male,27,24.15
0,3,male,28,7.86
1,3,female,23,8.05
0,3,male,30,8.66
1,2,female,22,21.0
0,3,male,46,7.2292
0,1,male,56,30.0
0,3,male,28,7.775
0,3,male,27,7.96
0,1,male,34,13.0
0,3,male,30,12.29
0,3,male,36,24.16
1,3,female,24,9.5
0,3,male,22,7.22
0,3,male,23,7.74
0,3,male,33,7.73
1,3,female,22,7.75
0,3,male,29,6.24
0,2,male,25,13.0
0,3,male,22,9.22
0,3,male,25,7.17
0,3,male,36,7.73
0,3,male,24,7.9
1,3,female,33,15.85
1,3,female,22,6.98
0,3,male,22,7.73
0,1,male,49,25.93
0,3,male,38,7.9
0,3,male,29,7.75
0,3,male,20,8.05
0,3,male,25,9.84
1,3,female,32,15.5
0,3,male,28,7.75
0,3,male,21,7.75
0,2,male,32,13.5
0,3,male,27,6.97
0,3,male,28,7.73
0,3,male,28,7.73
0,3,male,31,7.85
0,3,male,21,7.75
1,2,female,26,10.5
0,3,male,32,7.05
0,3,male,24,7.25
0,3,male,24,7.05
0,3,male,32,7.75
0,3,male,29,9.5
0,3,male,38,7.9
1,3,female,24,13.42
0,3,male,30,7.75
0,3,male,38,7.75
0,3,male,22,7.05
1,3,female,17,8.66
0,3,male,33,9.22
0,3,male,29,7.75
0,3,male,24,7.9
0,3,male,42,8.05
0,3,male,28,22.53
0,3,male,22,7.52
0,3,male,25,13.0
0,3,male,19,7.9
0,3,male,28,7.9
1,2,female,22,13.0
0,3,male,22,7.225
0,3,male,27,7.9
0,3,male,28,7.9
0,3,male,22,7.65
0,3,male,30,6.97
0,3,male,44,10.46
0,3,male,28,9.35
0,3,male,43,8.05
0,3,male,26,7.9
0,3,male,28,7.9
0,3,male,22,7.75
0,3,male,34,7.25
0,3,male,21,7.9
0,1,male,55,16.1
0,3,male,25,7.775
0,3,male,26,8.05
0,3,male,34,8.05
0,3,male,22,7.9
1,3,female,25,7.23
0,3,male,21,9.5
0,3,male,40,7.8958
0,3,male,34,7.8958
0,3,male,23,7.9
0,1,male,50,28.5
0,3,male,44,8.05
0,3,male,40,27.9
0,3,male,26,13.0
0,3,male,23,9.22
0,3,male,22,7.9
0,3,male,27,7.225
0,3,male,22,7.9
0,3,male,27,7.9
0,3,male,26,7.9
"""

df = pd.read_csv(io.StringIO(data_csv))
fitur = ["Pclass", "Sex", "Age", "Fare"]

# Encode fitur kategorikal: male=0, female=1
# (harus sama persis dengan cara app.py membaca input)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X, y = df[fitur], df["Survived"]

# Stratified split memastikan proporsi kelas seimbang di train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree dengan max_depth=4 untuk mencegah overfitting
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Hitung akurasi pada data uji (data yang tidak pernah dilihat model)
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi pada data uji: {accuracy:.2%}")

# Simpan model beserta nilai akurasi sebagai dictionary
# → app.py bisa menampilkan akurasi tanpa harus melatih ulang
joblib.dump({"model": model, "accuracy": round(accuracy, 4)}, "model.joblib")
print("model.joblib tersimpan.")
