# arima/app.py
# ============================================================
# Dataset  : Data pengguna Kereta Cepat Whoosh (Jakarta–Bandung)
#            Data sintetis deterministik untuk keperluan demonstrasi.
# Model    : ARIMA(1,1,1) — statsmodels
# Input    : Jumlah bulan yang ingin di-forecast (1–24)
# Output   : Tabel + chart data historis penumpang + prediksi
#
# Cara menjalankan:
#   1. python train.py   (sekali, untuk menghasilkan model.joblib)
#   2. python app.py
#   3. Buka http://127.0.0.1:5003
# ============================================================

from flask import Flask, render_template, request
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

app  = Flask(__name__)
data = joblib.load("model.joblib")
model_fit = data["model"]
series    = data["series"]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error  = None

    # Default: tampilkan 12 bulan historis terakhir sebagai chart awal (GET)
    # Ambil 24 data historis terakhir untuk ditampilkan di chart
    hist_dates  = [d.strftime("%b %Y") for d in series.index[-24:]]
    hist_values = [round(float(v), 1) for v in series.values[-24:]]

    n_forecast = 6  # default

    if request.method == "POST":
        try:
            n_forecast = int(request.form["n_forecast"])
            if not 1 <= n_forecast <= 24:
                raise ValueError("Jumlah bulan harus antara 1 dan 24.")

            # Jalankan forecast
            forecast_obj  = model_fit.forecast(steps=n_forecast)
            forecast_vals = [round(float(v), 1) for v in forecast_obj.values]

            # Buat label bulan prediksi (lanjutan dari data terakhir)
            last_date      = series.index[-1]
            forecast_dates = []
            cur = last_date
            for _ in range(n_forecast):
                # Majukan satu bulan
                month = cur.month % 12 + 1
                year  = cur.year + (1 if cur.month == 12 else 0)
                import datetime
                cur = datetime.date(year, month, 1).replace(
                    day=28  # aman untuk semua bulan
                )
                forecast_dates.append(
                    datetime.date(year, month, 1).strftime("%b %Y")
                )

            # Rangkuman statistik forecast
            avg_fc  = round(sum(forecast_vals) / len(forecast_vals), 1)
            max_fc  = max(forecast_vals)
            min_fc  = min(forecast_vals)
            trend   = "Naik" if forecast_vals[-1] > forecast_vals[0] else "Turun"

            result = {
                "n"              : n_forecast,
                "forecast_vals"  : forecast_vals,
                "forecast_dates" : forecast_dates,
                "avg"            : avg_fc,
                "max"            : max_fc,
                "min"            : min_fc,
                "trend"          : trend,
                # JSON untuk chart
                "chart_hist_labels"  : json.dumps(hist_dates),
                "chart_hist_vals"    : json.dumps(hist_values),
                "chart_fc_labels"    : json.dumps(forecast_dates),
                "chart_fc_vals"      : json.dumps(forecast_vals),
            }

        except ValueError as e:
            error = str(e)

    return render_template("index.html",
                           result=result,
                           error=error,
                           hist_dates=json.dumps(hist_dates),
                           hist_values=json.dumps(hist_values),
                           n_forecast=n_forecast)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
