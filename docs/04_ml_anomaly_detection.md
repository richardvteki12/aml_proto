# 04 — Unsupervised AML Anomaly Detection

Dokumen ini menjelaskan implementasi pada [04_ml_anomaly_detection.ipynb](../notebooks/04_ml_anomaly_detection.ipynb). Notebook membandingkan Isolation Forest dan Local Outlier Factor (LOF) untuk memberi ranking anomali transaksi AML synthetic, memilih konfigurasi terbaik melalui validation set, mengevaluasinya sekali pada test set yang belum disentuh, lalu menyimpan model untuk inference Streamlit.

> Penting: model dilatih tanpa scenario_id, known_aml_label, atau ground truth lain. Label synthetic hanya dipakai setelah model memberi score, yaitu untuk evaluasi ranking dan pemilihan hyperparameter.

> Scope typology: AML-S01 sampai AML-S05 — Structuring/Smurfing, Sudden Transaction Spike, Rapid Movement of Funds, Dormant Account Reactivation, dan Multiple Senders to One Receiver.

## 1. Ringkasan alur

    transaction_feature_abt.csv
              |
              | successful transactions only
              | exclude AML-S06 sampai AML-S10
              v
    temporal split: train -> validation -> untouched test
              |
              v
    feature contract + train-only preprocessing
              |
              +--> Isolation Forest baseline
              |
              +--> Local Outlier Factor baseline
                        |
                        v
          validation ranking metrics + ground truth
                        |
                        v
              tune only validation winner
                        |
                        v
          final evaluation on untouched test
                        |
                        +--> validation_locked_model.joblib
                        |
                        +--> refit train + validation
                                  |
                                  v
                        best_anomaly_model.joblib for Streamlit

Notebook ini tidak menggantikan rule-based monitoring. Rule dan model berjalan sebagai dua sinyal terpisah:

- Rule-based menjelaskan red flag yang telah diketahui dan mempunyai threshold eksplisit.
- ML anomaly detection memberi ranking untuk kombinasi perilaku yang tidak selalu tepat memenuhi threshold rule.

Dalam eksperimen, any_rule_alert hanya dipakai untuk metrik rule-miss recovery sesudah scoring. Candidate flag rule tidak menjadi input model, sehingga hasil ML tidak sekadar meniru rule sebelumnya.

## 2. Tujuan model, populasi, dan batasan

### 2.1 Tujuan model

Tujuan model bukan memutuskan bahwa transaksi adalah TPPU. Model mengurutkan transaksi berdasarkan tingkat ketidaklaziman relatif terhadap perilaku data training.

    anomaly_score lebih besar = transaksi lebih tidak lazim menurut model

Score bukan probabilitas dan tidak memiliki skala universal seperti 0–1 atau 0–100. Nilai score terutama dipakai untuk ranking dalam batch dan dibandingkan hanya dengan score dari model/artifact yang sama.

### 2.2 Populasi yang masuk model

Model data dibentuk dari Feature ABT dengan aturan berikut:

1. Hanya transaksi dengan is_success = True.
2. Semua transaksi yang berada pada ground truth AML-S06 sampai AML-S10 dikeluarkan.
3. Ground truth AML-S01 sampai AML-S05 ditempelkan hanya sebagai evaluation metadata.

Lima typology pertama sudah memiliki behavioural feature eksplisit di ABT. Menggabungkan scenario di luar scope sebagai baseline atau positive akan membuat evaluasi tidak jelas karena feature untuk scenario tersebut belum dibangun pada scope ini.

| Kondisi | known_aml_label |
|---|---:|
| transaction_id muncul pada ground truth AML-S01 sampai AML-S05 | 1 |
| Baris lain dalam populasi eksperimen | 0 |

Nilai 0 berarti synthetic baseline proxy, bukan bukti bahwa transaksi pasti normal di dunia nyata. Pada data produksi, label positif biasanya lebih terbatas, tertunda, dan hasil investigasi dapat berubah.

## 3. Input, output, dan dependency

### 3.1 Input utama

| Artefak | Fungsi |
|---|---|
| data/processed/transaction_feature_abt.csv | Input Feature ABT dari Notebook 03, satu baris per transaksi |
| data/ground_truth/aml_ground_truth.csv | Label synthetic untuk evaluasi saja |
| src/aml_ml_features.py | Kontrak feature, preprocessing, dan class model bundle untuk inference |

ABT harus memenuhi kontrak berikut sebelum training dimulai:

- transaction_id unik;
- transaction_timestamp ter-parse dan tidak kosong;
- seluruh transaction_id pada ground truth tersedia tepat satu kali di ABT; dan
- seluruh feature dalam src/aml_ml_features.py tersedia.

### 3.2 Dependency Python

Notebook memakai pandas, numpy, scikit-learn, joblib, dan IPython.

Model saat ini memakai scikit-learn standar. Tidak ada library CUDA atau cuML di code, sehingga training berjalan di CPU. Menjalankan notebook di Google Colab runtime GPU tidak otomatis membuat Isolation Forest atau LOF memakai GPU; pipeline harus diubah ke library yang memang mendukung GPU terlebih dahulu.

### 3.3 Output yang disimpan

    data/processed/ml_anomaly_detection/
    ├── baseline_model_comparison.csv
    ├── winner_hyperparameter_tuning.csv
    ├── validation_scored_transactions.csv
    ├── test_scored_transactions.csv
    └── test_recall_by_typology.csv

    models/aml_anomaly_detection/
    ├── validation_locked_model.joblib
    ├── best_anomaly_model.joblib
    ├── model_feature_schema.json
    └── model_metadata.json

## 4. Konfigurasi eksperimen yang terlihat

| Parameter | Nilai implementasi saat ini | Arti |
|---|---:|---|
| RANDOM_STATE | 42 | Membuat Isolation Forest reproducible |
| TRAIN_END | 2026-03-27 00:00:00 exclusive | Batas akhir data train |
| VALIDATION_END | 2026-05-14 00:00:00 exclusive | Batas akhir validation; sesudahnya adalah test |
| TOP_FRACTION | 0,01 | Budget review: 1% transaksi dengan score tertinggi |
| LOF_REFERENCE_SAMPLE_SIZE | 15.000 | Jumlah row train yang menjadi reference neighbourhood LOF |
| LOF_ALGORITHM | ball_tree | Algoritma pencarian neighbour LOF |
| LOF_N_JOBS | 1 | Setting stabil untuk Windows |

### 4.1 Catatan penting tentang LOF reference sample

Code aktif menetapkan LOF_REFERENCE_SAMPLE_SIZE = 15000. Maka LOF tidak dilatih pada seluruh train set saat ini. Ia memakai 15.000 row reference yang dipilih secara deterministik dan tersebar merata sepanjang timeline train.

Jika parameter diubah menjadi None, fungsi temporal_reference_indices mengembalikan seluruh indeks train dan LOF akan memakai full train set.

Ada keterangan Markdown lama di notebook yang menjelaskan mode None atau full train. Keterangan itu menjelaskan opsi konfigurasi, bukan nilai code aktif saat ini. Dokumentasi ini mengikuti nilai executable yang sebenarnya: 15.000 row.

Sampling tidak memakai label, rule alert, atau score untuk memilih row. Tujuannya hanya membatasi biaya komputasi LOF yang distance-based. Isolation Forest selalu memakai seluruh train matrix.

## 5. Temporal split dan pencegahan leakage

| Split | Kondisi waktu | Peran |
|---|---|---|
| Train | timestamp sebelum 2026-03-27 | Fit preprocessor dan dua baseline model |
| Validation | 2026-03-27 sampai sebelum 2026-05-14 | Memilih baseline winner dan hyperparameter |
| Test | timestamp pada/di atas 2026-05-14 | Evaluasi final satu kali, tidak ikut memilih model |

Temporal split lebih realistis untuk AML monitoring: model belajar dari masa lalu lalu mengevaluasi transaksi yang datang kemudian. Mengacak waktu berisiko membuat distribusi masa depan ikut memengaruhi training.

### 5.1 Scenario group safety check

Beberapa synthetic typology adalah pola multi-transaksi. Notebook memeriksa bahwa satu scenario_group_id tidak terpecah antara train, validation, dan test.

    jumlah split unik per scenario_group_id harus kurang dari atau sama dengan 1

Tanpa check ini, model bisa melihat sebagian transaksi dari satu pattern di training lalu dinilai pada sisa pattern yang sama di test. Itu adalah leakage dan akan membuat metrik terlalu optimistis.

### 5.2 Ground truth dipisahkan dari feature matrix

Ground truth dibaca ke DataFrame terpisah. Setelah label ditempelkan untuk evaluation metadata, fungsi pembentuk feature tetap tidak menerima label tersebut. Assertion memastikan:

- lima candidate flag rule tidak ada pada model input;
- known_aml_label tidak ada pada model input; dan
- index model input tetap sejajar dengan model_data.

## 6. Kontrak feature model

Kontrak feature berada di src/aml_ml_features.py, bukan tersembunyi di dalam estimator. Pemisahan ini penting karena Streamlit nanti mengirim DataFrame berbentuk ABT, bukan NumPy matrix yang sudah diproses.

Ada 40 raw ABT feature yang dibaca. transaction_hour diubah menjadi dua feature siklikal, sehingga terdapat 41 kolom model input sebelum one-hot encoding. One-hot encoding kemudian dapat menambah jumlah kolom final sesuai kategori yang terlihat pada data train.

### 6.1 Numeric: log1p dan RobustScaler (22)

| No. | Feature | Intuisi |
|---:|---|---|
| 1 | amount_idr_equivalent | Besar nominal transaksi saat ini |
| 2 | sender_customer_monthly_income | Konteks kemampuan ekonomi sender |
| 3 | sender_success_txn_count_1h | Velocity sender satu jam |
| 4 | sender_success_amount_sum_1h_idr | Total dana sender satu jam |
| 5 | sender_success_txn_count_120m | Velocity sender dua jam |
| 6 | sender_success_amount_sum_120m_idr | Total dana sender dua jam |
| 7 | sender_success_txn_count_24h | Velocity sender 24 jam |
| 8 | sender_success_amount_sum_24h_idr | Total dana sender 24 jam |
| 9 | sender_success_txn_count_7d | Velocity sender tujuh hari |
| 10 | sender_success_amount_sum_7d_idr | Total dana sender tujuh hari |
| 11 | sender_subthreshold_txn_count_24h | Jumlah transaksi kecil sender 24 jam |
| 12 | sender_subthreshold_amount_sum_24h_idr | Total transaksi kecil sender 24 jam |
| 13 | minutes_since_last_internal_inbound | Jeda inbound internal terakhir ke outbound sekarang |
| 14 | outbound_to_last_inbound_ratio | Perbandingan nominal outbound dan inbound terakhir |
| 15 | prior_success_txn_count_30d | Jumlah aktivitas sukses sender sebelum event, 30 hari |
| 16 | amount_to_prior_median_ratio_30d | Nominal sekarang relatif terhadap median historis |
| 17 | days_since_prior_successful_sender_activity | Jeda hari sejak aktivitas sukses sender sebelumnya |
| 18 | receiver_txn_count_24h | Banyak transaksi yang diterima receiver party, 24 jam |
| 19 | distinct_senders_to_receiver_24h | Banyak sender berbeda ke receiver party, 24 jam |
| 20 | receiver_amount_sum_24h_idr | Total dana diterima receiver party, 24 jam |
| 21 | receiver_txn_count_7d | Banyak transaksi receiver party, tujuh hari |
| 22 | distinct_senders_to_receiver_7d | Banyak sender berbeda ke receiver party, tujuh hari |

log1p menekan distribusi nominal, count, dan ratio yang berekor panjang. RobustScaler memakai statistik yang lebih tahan terhadap outlier dibanding mean atau standard deviation biasa. Hal ini terutama penting untuk LOF karena jarak antar-point akan terdistorsi jika satu nominal Rupiah mendominasi seluruh feature lain.

### 6.2 Binary context (5)

| Feature | Arti |
|---|---|
| has_prior_internal_inbound_24h | Ada inbound internal sebelumnya dalam 24 jam |
| has_sufficient_history_30d | Riwayat 30 hari sender cukup untuk baseline spike |
| has_prior_successful_sender_activity | Sender punya aktivitas sukses sebelumnya |
| sender_customer_pep_flag | Sender ditandai PEP pada master synthetic |
| is_internal_receiver | Penerima adalah customer internal |

Feature binary tidak diberi transformasi log; preprocessing hanya melindungi terhadap missing value dengan most-frequent imputation.

### 6.3 Waktu siklikal (2)

| Feature | Rumus | Mengapa |
|---|---|---|
| transaction_hour_sin | sin(2 pi x hour / 24) | Menjaga kedekatan jam 23 dan jam 00 |
| transaction_hour_cos | cos(2 pi x hour / 24) | Pasangan koordinat siklikal untuk waktu 24 jam |

Menggunakan transaction_hour sebagai angka mentah akan memperlakukan jam 23 dan 0 sebagai sangat jauh, padahal keduanya bersebelahan.

### 6.4 Categorical: one-hot encoded (12)

| Feature | Arti |
|---|---|
| transaction_type | Transfer, Cash, RTGS, SWIFT, BI-FAST, dan sebagainya |
| channel | Mobile, Internet, Branch, ATM, API |
| currency | Mata uang transaksi |
| purpose_code | Kode tujuan transaksi |
| source_of_fund | Sumber dana yang dicatat |
| destination_country | Negara tujuan |
| sender_customer_segment | Segmentasi customer sender |
| sender_customer_risk_rating | Risk rating customer sender |
| sender_account_type | Jenis rekening sender |
| sender_account_risk_level | Risk level rekening sender |
| receiver_party_country | Negara receiver party conformed |
| receiver_party_risk_level | Risk level receiver party conformed |

One-hot encoding mengubah kategori menjadi kolom biner. handle_unknown = ignore memastikan kategori baru pada data inference tidak membuat Streamlit error; kategori tersebut hanya tidak menyalakan kolom kategori yang telah dikenal model.

### 6.5 Feature yang sengaja dikeluarkan

Model tidak menerima:

- ID, nama, alamat, nomor account, dan atribut identitas lain;
- raw transaction_timestamp;
- scenario_id, scenario_name, scenario_group_id, known_aml_label, atau field ground truth lain;
- lima candidate flag rule; dan
- debit_credit.

Alasannya adalah menghindari memorisasi identitas, menghindari leakage, serta menjaga perbandingan rule versus ML tetap adil. Model harus belajar dari perilaku dan konteks, bukan dari jawaban synthetic atau transaksi yang sudah dinyatakan alert oleh rule.

## 7. Missing-value handling dan preprocessing

### 7.1 Sentinel history menjadi missing yang jujur

Pada Feature ABT, nilai -1 berarti tidak ada riwayat sebelumnya, bukan nilai durasi negatif. Sebelum model diproses, dua kolom ini mengubah -1 menjadi NaN:

    minutes_since_last_internal_inbound
    days_since_prior_successful_sender_activity

Informasi adanya atau tidak adanya history tidak hilang karena ada flag pendamping:

- has_prior_internal_inbound_24h; dan
- has_prior_successful_sender_activity.

### 7.2 Train-only preprocessing

Preprocessor di-fit hanya pada X_train_raw.

| Keluarga feature | Pipeline | Perlakuan missing |
|---|---|---|
| Numeric | median imputation + missing indicator -> log1p -> RobustScaler | Median dari train dan indicator availability |
| Binary | most-frequent imputation | Nilai paling sering dari train |
| Cyclical time | median imputation -> RobustScaler | Median dari train |
| Categorical | constant imputation -> one-hot | __MISSING__ dari train |

Validation dan test hanya menjalankan transform. Tidak ada imputer, scaler, atau encoder yang belajar dari masa depan. Setelah transform, notebook meng-assert bahwa matrix train, validation, dan test tidak mengandung NaN atau inf.

## 8. Baseline model

### 8.1 Isolation Forest

Isolation Forest mencoba mengisolasi observasi dengan pemisahan acak berbasis pohon. Observasi yang lebih mudah diisolasi dipandang lebih anomali.

| Parameter | Nilai |
|---|---:|
| n_estimators | 300 |
| max_samples | 256 |
| max_features | 1,0 |
| bootstrap | False |
| contamination | auto |
| n_jobs | -1 |

Isolation Forest di-fit menggunakan seluruh X_train.

### 8.2 Local Outlier Factor

LOF membandingkan kepadatan lokal sebuah transaksi dengan kepadatan tetangganya. Transaksi di daerah yang jauh lebih jarang daripada tetangga lokalnya akan memperoleh sinyal anomali yang lebih tinggi.

| Parameter | Nilai |
|---|---:|
| n_neighbors | 35 |
| algorithm | ball_tree |
| leaf_size | 40 |
| contamination | auto |
| novelty | True |
| n_jobs | 1 |

novelty = True wajib karena model perlu memberi score pada validation dan test yang tidak dipakai sebagai reference fit. Dengan novelty = False, LOF ditujukan hanya untuk deteksi outlier pada data yang sama dengan data fit dan tidak sesuai untuk alur inference ini.

### 8.3 Orientasi anomaly score

Kedua estimator scikit-learn mengeluarkan raw score yang lebih rendah untuk row yang lebih unusual. Notebook membalik tanda satu kali melalui fungsi berikut:

    anomaly_score = -estimator.score_samples(transformed_matrix)

Dengan kontrak ini, semua output mengikuti aturan sederhana: semakin besar anomaly_score, semakin tinggi prioritas review.

Nilai seperti 1,10, 1,50, atau 2,00 bukan persentase. Tidak tepat membaca 1,50 sebagai 150% AML atau dua kali lebih berbahaya dari 0,75. Gunakan urutan/rank, Top-K, dan distribusi score pada batch yang sama.

## 9. Evaluasi baseline pada validation set

### 9.1 Mengapa evaluasi berbasis ranking

Dalam monitoring AML, tim investigasi tidak meninjau seluruh transaksi. TOP_FRACTION = 0,01 mensimulasikan kapasitas untuk meninjau 1% transaksi dengan anomaly score tertinggi. Karena itu, model dibandingkan sebagai alat ranking, bukan hanya memakai binary prediction dari contamination.

### 9.2 Metrik

| Metrik | Makna |
|---|---|
| roc_auc | Kemampuan umum mengurutkan positive di atas baseline proxy; metrik sekunder |
| average_precision | Kualitas ranking pada kelas positive yang jarang; lebih informatif daripada accuracy pada dataset imbalanced |
| random_precision_baseline | Positive rate populasi; pembanding AP acak |
| average_precision_lift_vs_random | average_precision dibagi random_precision_baseline |
| top_k_rows | Jumlah transaksi pada Top 1% |
| precision_at_top_k | Positive synthetic dalam Top-K dibagi seluruh Top-K |
| recall_at_top_k | Positive synthetic yang masuk Top-K dibagi seluruh positive synthetic |
| rule_miss_recovery_at_top_k | Bagian positive yang tidak dipicu lima rule tetapi masuk Top-K ML |

Urutan pemilihan baseline winner adalah:

    Average Precision  ->  Recall at Top-1%  ->  ROC-AUC

Artinya, apabila dua model memiliki Average Precision sama, recall pada budget review menjadi pembanding berikutnya; ROC-AUC menjadi tie-breaker terakhir.

### 9.3 Recall per typology

Notebook juga melaporkan recall_at_top_k terpisah untuk AML-S01 sampai AML-S05. Metrik ini menjawab pertanyaan yang lebih operasional: pola mana yang benar-benar terangkat ke antrean review teratas?

Top-K adalah Top-K global dalam satu split, bukan Top-K terpisah untuk setiap typology. Karena itu, recall per typology memperlihatkan persaingan nyata antar-typology untuk kapasitas review yang sama.

## 10. Hyperparameter tuning pada unsupervised model

Unsupervised model tetap dapat dituning. Yang membuat training tetap unsupervised adalah fit tidak menerima target label. Label hanya dipakai setelah setiap candidate selesai scoring validation untuk memilih konfigurasi yang memberi ranking investigasi terbaik.

Ini berbeda dari GridSearchCV supervised yang mengoptimalkan accuracy atau loss langsung pada label ketika fitting.

### 10.1 Candidate grid

Jika Isolation Forest menjadi baseline winner, kandidatnya adalah:

| Kandidat | n_estimators | max_samples | max_features |
|---|---:|---:|---:|
| 1 | 300 | 256 | 1,0 |
| 2 | 500 | 256 | 1,0 |
| 3 | 500 | 512 | 1,0 |
| 4 | 500 | 512 | 0,7 |

Jika LOF menjadi baseline winner, kandidatnya adalah n_neighbors = 20, 35, dan 50; parameter lain tetap ball_tree, leaf_size = 40, dan contamination = auto.

Semua candidate memakai preprocessing yang sudah di-fit pada train dan hanya dinilai pada validation. Test set tidak dipakai untuk memilih algorithm, parameter, feature set, atau policy Top-K.

## 11. Final test evaluation

Model terpilih untuk report final adalah estimator yang:

1. di-fit pada train;
2. dipilih melalui validation; lalu
3. memberi score pada test yang belum pernah dilihat untuk selection.

Test output menyimpan:

| Kolom | Arti |
|---|---|
| transaction_id, transaction_timestamp | Identitas dan waktu transaksi |
| scenario_id, scenario_name, scenario_group_id | Metadata ground truth untuk evaluasi |
| known_aml_label | Label evaluation-only |
| any_rule_alert | Apakah salah satu dari lima rule flag aktif |
| anomaly_score | Score yang lebih tinggi berarti lebih anomalous |
| is_top_k_alert | 1 jika masuk Top 1% global test batch |
| anomaly_rank | Rank unik; 1 adalah transaksi paling anomalous |

test_recall_by_typology.csv menambahkan jumlah known AML, jumlah yang masuk Top-K, recall per typology, serta rata-rata score per typology.

## 12. Artefak model untuk inference

### 12.1 validation_locked_model.joblib

Artifact ini berisi model yang hanya di-fit pada train. Ia menyimpan reference_threshold dari quantile 99% anomaly score validation dan dipakai untuk menjaga hasil test report dapat direproduksi.

Threshold tersebut tidak otomatis dipindahkan ke production model karena score distribution dapat berubah sesudah refit menggunakan data train + validation.

### 12.2 best_anomaly_model.joblib

Ini artifact production untuk Streamlit. Setelah model terbaik dipilih, preprocessor baru dan estimator baru di-fit ulang menggunakan gabungan train + validation, tetap tanpa ground truth label.

Artifact menyimpan:

- nama model;
- estimator ter-fit;
- preprocessor ter-fit;
- review_top_fraction = 0,01;
- feature contract; dan
- catatan training.

reference_threshold = None pada artifact production adalah pilihan yang disengaja. Streamlit harus:

1. menerima batch transaksi ABT baru;
2. memanggil bundle.anomaly_score(abt_dataframe);
3. mengurutkan score secara descending; dan
4. mengirim Top 1% batch ke review.

Policy ranking ini lebih stabil daripada memakai angka score absolut dari model yang sudah di-refit.

### 12.3 Mengapa src/aml_ml_features.py harus ikut deployment

Joblib perlu menemukan class AMLAnomalyScoringBundle ketika menjalankan joblib.load. Selain itu, class tersebut menjalankan kontrak feature dan preprocessor yang sama saat inference.

Deployment Streamlit minimal harus menyertakan:

    models/aml_anomaly_detection/best_anomaly_model.joblib
    src/aml_ml_features.py
    dataframe ABT dengan seluruh raw feature contract

## 13. Inference smoke test

Di akhir notebook, model production dimuat ulang dari file, bukan memakai object Python yang masih ada di memory. Lima transaksi awal dari test set diberikan kepada model sebagai simulasi batch inference yang sebelumnya tidak masuk training production.

Smoke test memeriksa:

- jumlah score sama dengan jumlah input;
- seluruh score adalah finite, bukan NaN atau inf; dan
- batch dapat diranking dari score terbesar ke terkecil.

Smoke test membuktikan artifact dapat dimuat serta memberi score. Ia belum menggantikan integration test Streamlit atau monitoring data drift di production.

## 14. Snapshot artefak eksperimen yang tersimpan saat ini

Tabel berikut berasal dari file hasil yang saat ini tersimpan di project. Nilai dapat berubah bila data, ABT, konfigurasi, atau model dijalankan ulang; jangan menganggapnya sebagai target tetap.

### 14.1 Baseline validation

| Model | ROC-AUC | Average Precision | Recall at Top-1% | Precision at Top-1% | Fit time |
|---|---:|---:|---:|---:|---:|
| Local Outlier Factor, n_neighbors = 35 | 0,9087 | 0,3122 | 49,09% | 11,09% | 60,71 detik |
| Isolation Forest | 0,8935 | 0,1388 | 31,82% | 7,19% | 2,18 detik |

LOF menjadi baseline winner karena Average Precision validation lebih tinggi.

### 14.2 Hasil tuning winner

LOF dengan n_neighbors = 50 dipilih. Pada validation, kandidat ini menghasilkan Average Precision 0,3265, ROC-AUC 0,9131, Recall at Top-1% 51,82%, dan Precision at Top-1% 11,70%.

### 14.3 Final test yang belum disentuh saat selection

| Metrik | Nilai tersimpan |
|---|---:|
| Test rows | 48.621 |
| Known AML positive | 85 |
| ROC-AUC | 0,9070 |
| Average Precision | 0,3125 |
| AP lift dibanding random | 178,8x |
| Top-1% review rows | 487 |
| Known AML di Top-1% | 48 |
| Recall at Top-1% | 56,47% |
| Precision at Top-1% | 9,86% |
| Rule-missed known AML | 24 |
| Rule misses yang direcover ML di Top-1% | 3, yaitu 12,5% |

Per typology pada saved test output: Rapid Movement memiliki recall Top-1% 100%, Dormant Reactivation 90%, Structuring 40%, Spike 11,11%, dan Multiple Senders 6,25%. Perbedaan ini menunjukkan bahwa model tidak sama kuatnya pada seluruh pola dan rule tetap diperlukan sebagai komponen hybrid.

## 15. Interpretasi untuk presentasi

Cara menjelaskan hasil ini ke stakeholder:

1. Model tidak diberi jawaban saat training. Ia belajar struktur perilaku dari feature ABT; label hanya mengecek apakah ranking yang dihasilkan berguna.
2. Top-1% adalah kapasitas review, bukan threshold regulasi. Dari 48.621 transaksi test, 487 transaksi teratas dibawa ke investigasi.
3. Average Precision jauh di atas baseline random berarti known synthetic AML lebih terkonsentrasi di ranking tinggi daripada pemilihan acak.
4. ML melengkapi rule. Sebagian case yang tidak memicu lima rule masih masuk antrean Top-K ML, tetapi rule tetap lebih explainable untuk red flag eksplisit.
5. Anomaly score bukan probability. Analyst harus melihat ranking, feature behaviour, rule alert, dan konteks KYC sebelum mengambil tindakan.

## 16. Limitasi dan penggunaan yang aman

- Dataset dan ground truth bersifat synthetic; metrik tidak boleh dianggap sebagai performa production sesungguhnya.
- LOF lebih mahal daripada Isolation Forest. Dengan reference 15.000 row, saved run membutuhkan puluhan detik untuk fit; full train set akan lebih berat secara memori dan waktu.
- Feature/category distribution dapat berubah saat data baru masuk. Batch ranking membantu operasi, tetapi tetap perlu monitoring drift dan kalibrasi ulang.
- contamination = auto bukan policy alert operational. Policy review eksplisit di sini adalah Top 1%.
- Model unsupervised tidak menyediakan reason code seperti rule secara otomatis. UI hybrid sebaiknya menampilkan anomaly rank bersama feature/rule evidence yang relevan.
- Jangan gunakan known_aml_label, scenario_id, atau candidate flag rule sebagai model input pada eksperimen lanjutan tanpa menyatakan perubahan desain secara eksplisit.

## 17. Catatan housekeeping notebook

Nama file yang dipakai adalah 04_ml_anomaly_detection.ipynb, tetapi heading internal notebook masih berbunyi 05 — Unsupervised AML Anomaly Detection dan terdapat satu Markdown cell berisi traceback encoding lama. Keduanya tidak memengaruhi model calculation, tetapi sebaiknya dirapikan saat maintenance agar nomor dan narasi tampil konsisten.

## 18. Troubleshooting

| Gejala | Kemungkinan penyebab | Tindakan |
|---|---|---|
| Project root tidak ditemukan | Feature ABT belum ada atau kernel dijalankan di luar project | Jalankan Notebook 03 dan buka notebook dari root/notebooks project |
| KeyError feature contract | ABT berubah atau belum lengkap | Bandingkan header ABT dengan RAW_FEATURE_COLUMNS pada src/aml_ml_features.py |
| Matrix masih NaN atau inf | Feature baru/sentinel belum ditangani preprocessor | Periksa build_model_input dan pipeline imputation sebelum fit ulang |
| LOF lambat atau kehabisan RAM | Reference sample terlalu besar atau feature/category bertambah | Pertahankan/kurangi LOF_REFERENCE_SAMPLE_SIZE, atau pilih Isolation Forest bila latency lebih penting |
| Ingin LOF full train set | LOF_REFERENCE_SAMPLE_SIZE masih integer | Ubah menjadi None, pertimbangkan RAM/waktu, lalu jalankan eksperimen ulang |
| Score production berbeda dari score validation | Production model di-refit pada train + validation | Ini normal; gunakan ranking Top-1%, bukan validation threshold absolut |
| joblib.load gagal di Streamlit | Modul/class inference tidak ikut deployment | Sertakan src/aml_ml_features.py, dependency sklearn/joblib kompatibel, dan artifact model |
| Metrik test terlalu bagus | Ground truth atau group mungkin bocor ke training/selection | Periksa temporal split, scenario_group_id assertion, dan pastikan test tidak dipakai saat tuning |

## 19. Checklist handoff ke Streamlit

- [ ] Notebook 03 menghasilkan ABT sesuai feature contract.
- [ ] Model final dipilih dengan validation, bukan test.
- [ ] best_anomaly_model.joblib dan src/aml_ml_features.py tersedia.
- [ ] UI memberi rank/Top-1% batch, bukan menafsirkan anomaly score sebagai probability.
- [ ] UI menampilkan anomaly score bersama rule alert dan evidence feature agar investigator dapat memahami prioritas.
- [ ] Ground truth tidak ikut dikirim sebagai input inference.
