# AML Detection Studio

Next.js UI lokal untuk dua kegunaan:

1. **Page 1 — Evaluasi:** menampilkan ground truth, hasil 5 rule AML monitoring, perbandingan Isolation Forest vs Local Outlier Factor, dan final holdout metrics dari artefak proyek.
2. **Page 2 — Inference:** menerima transaksi baru atau preset tipologi, lalu menampilkan rule hit dan anomaly score dari model yang sudah tersimpan.

## Menjalankan aplikasi

Jalankan dari folder `web`:

```powershell
cd "E:\Trading\V-Teki Project\aml_proto\web"
$env:AML_PYTHON_EXECUTABLE = "E:\Anaconda3\envs\super\python.exe"
npm run dev
```

Buka `http://localhost:3000`.

`AML_PYTHON_EXECUTABLE` diperlukan karena artefak model adalah bundle scikit-learn/Joblib. Route Next.js memanggil Python untuk memuat dan melakukan scoring terhadap `models/aml_anomaly_detection/best_anomaly_model.joblib`; tidak ada proses `fit` atau training ulang pada inference.

## Saat data evaluasi berubah

Halaman evaluasi membaca snapshot kecil di `public/dashboard-data.json`. Bangun ulang snapshot setelah mengganti ABT, ground truth, atau artefak evaluasi:

```powershell
npm run refresh:dashboard
```

Perintah tersebut hanya membaca artefak yang sudah ada dan memperbarui JSON dashboard. Ia tidak melatih ulang model.

## Validasi lokal

```powershell
npm run lint
npm run build
```

Untuk menjalankan versi production lokal:

```powershell
npm run start
```

## Batasan interpretasi

- Rule adalah candidate/red flag untuk investigasi, bukan kesimpulan hukum AML.
- `anomaly_score` adalah skor ranking: semakin besar, semakin tidak lazim. Itu bukan probabilitas AML.
- Model saat ini berada dalam scope transaksi berhasil dan lima tipologi AML-S01 sampai AML-S05.
- Model menggunakan kebijakan review top 1% **dalam satu batch**; persentil yang tampil di Page 2 hanya kalibrasi terhadap holdout reference agar satu transaksi mudah dibaca.
