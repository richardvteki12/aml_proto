# 03 — Raw Join, Feature Engineering, dan Feature ABT

Dokumen ini menjelaskan notebook [`03_feature_engineering.ipynb`](../notebooks/03_feature_engineering.ipynb) berdasarkan implementasinya saat ini. Notebook mengambil CSV synthetic AML, memuatnya ke PostgreSQL, menjahit empat tabel inti menjadi satu tabel ber-grain transaksi, membangun feature leakage-safe untuk lima tipologi AML, lalu menyimpan Feature ABT.

> **Output utama:** [`data/processed/transaction_feature_abt.csv`](../data/processed/transaction_feature_abt.csv), dengan satu baris untuk satu `transaction_id`.

> **Scope saat ini:** lima tipologi: Structuring/Smurfing, Rapid Movement of Funds, Sudden Transaction Spike, Dormant Account Reactivation, dan Multiple Senders to One Receiver. `sanctions_watchlist` dimuat ke PostgreSQL, tetapi belum dijahit menjadi feature di notebook ini.

## 1. Tujuan notebook dan batasannya

Notebook ini memiliki empat tujuan berurutan:

1. membuat/mengecek database PostgreSQL `aml` dan schema `raw`;
2. mengimpor CSV synthetic ke tabel `raw`;
3. melakukan join empat tabel inti dengan SQL sehingga satu baris tetap mewakili satu transaksi; dan
4. membuat feature historis dan candidate flag untuk lima pola AML.

Notebook ini **bukan** tempat untuk:

- membuat label AML baru;
- menggunakan ground truth sebagai input feature;
- membangun 20 AML monitoring rule;
- melatih model machine learning; atau
- membuat sanctions screening feature.

Di bagian paling bawah terdapat evaluasi ringkas lima candidate flag terhadap ground truth. Bagian itu mengevaluasi hasil feature; ia tidak membuat feature baru dan tidak mengubah ABT.

## 2. Alur data dan grain

```text
01_generate_synthetic_data.ipynb
      |
      +--> data/raw/*.csv
                |
                v
        PostgreSQL: raw.accounts, raw.customers,
        raw.counterparties, raw.transactions,
        raw.sanctions_watchlist
                |
                v
        sql/join_four_tables.sql
                |
                v
        joined_tables  [1 row = 1 transaction_id]
                |
                v
        standardisasi + historical feature engineering
                |
                v
        data/processed/transaction_feature_abt.csv
                |
                +--> Rule-based evaluation (hanya membaca ABT)
                +--> Model anomaly detection pada notebook berikutnya
```

Grain tidak boleh berubah selama notebook berjalan:

```text
Satu baris Feature ABT = satu transaction_id = satu kejadian transaksi
```

Seluruh agregasi historis—misalnya jumlah transaksi dalam 24 jam—ditambahkan sebagai **kolom pada transaksi saat ini**. Feature tersebut tidak mengubah tabel menjadi satu baris per customer, account, atau hari.

## 3. Prasyarat dan cara menjalankan

### 3.1 Data dan aplikasi yang dibutuhkan

Sebelum menjalankan notebook, pastikan hal berikut tersedia.

| Kebutuhan | Alasan |
|---|---|
| Output dari `01_generate_synthetic_data.ipynb` | Menyediakan lima raw CSV serta ground truth |
| PostgreSQL aktif di `localhost:5432` | Database `aml` dan schema `raw` dibuat/diakses di sana |
| Role PostgreSQL `postgres` dan password yang benar | Notebook meminta password melalui `getpass()` |
| Paket `psycopg` | Koneksi PostgreSQL dan pembuatan database |
| Paket `sqlalchemy` | Import DataFrame ke PostgreSQL dan query SQL |
| Paket `pandas`, `numpy` | Data preparation serta feature calculation |
| Paket `rapidfuzz` | Diimpor pada setup saat ini, walaupun belum digunakan oleh feature aktif |
| File [`sql/join_four_tables.sql`](../sql/join_four_tables.sql) | SQL yang menghasilkan base table |

Pada cell awal, notebook menjalankan `%pip install psycopg` dan `%pip install sqlalchemy`. Bila package sudah tersedia pada kernel yang benar, cell tersebut hanya memastikan dependency ada. Setelah instalasi, restart kernel bila Jupyter meminta agar import menggunakan environment yang baru.

### 3.2 Urutan menjalankan notebook

1. Jalankan Notebook 01 terlebih dahulu dan pastikan CSV synthetic tersedia di `data/raw`.
2. Buka Notebook 03 dari folder `notebooks`.
3. Jalankan section setup dan masukkan password PostgreSQL ketika diminta.
4. Jalankan import raw table sampai `load_summary_df` tampil.
5. Pastikan `sql/join_four_tables.sql` ada, lalu jalankan join dan inspeksi hasilnya.
6. Jalankan standardisasi, seluruh feature section, dan bagian save ABT **secara berurutan**.
7. Pastikan semua assertion berhasil dan file `data/processed/transaction_feature_abt.csv` terbentuk.
8. Jalankan evaluasi rule-based hanya setelah ABT selesai ditulis.

Notebook mengandalkan object di memory dari cell sebelumnya, misalnya `joined_tables`, `base_transactions`, dan `transaction_feature_abt`. Karena itu, menjalankan cell di tengah tanpa setup sebelumnya dapat menghasilkan `NameError` atau output yang tidak konsisten. Untuk hasil yang repeatable, gunakan **Run All** setelah seluruh konfigurasi sudah benar.

### 3.3 Catatan path project

Pada setup awal, notebook berusaha menemukan project root dari current working directory. Namun, cell join dan save juga memakai path eksplisit:

```text
E:\Trading\V-Teki Project\Anti Money Laundering Detection Updated
```

Artinya, versi notebook sekarang memang ditujukan untuk folder tersebut. Bila project dipindahkan, `PROJECT_ROOT` pada cell join perlu diubah terlebih dahulu agar SQL dan output CSV ditemukan/ditulis ke lokasi yang benar.

## 4. Load raw CSV ke PostgreSQL

### 4.1 Pembuatan database dan schema

Notebook lebih dulu terhubung ke database bawaan PostgreSQL bernama `postgres` menggunakan `autocommit=True`. Autocommit diperlukan karena perintah `CREATE DATABASE` tidak boleh berjalan dalam transaksi biasa.

Jika database `aml` belum ada, notebook membuatnya dengan encoding UTF-8. Bila sudah ada, database tidak dihapus dan tidak dibuat ulang. Setelah koneksi SQLAlchemy terbentuk, notebook membuat schema berikut bila belum tersedia:

```sql
CREATE SCHEMA IF NOT EXISTS raw;
```

### 4.2 Tabel yang di-load

| CSV | Target PostgreSQL | Date parsing pada Pandas | Dipakai pada join/feature saat ini? |
|---|---|---|---|
| `accounts.csv` | `raw.accounts` | `opening_date` | Ya |
| `counterparties.csv` | `raw.counterparties` | Tidak ada | Ya |
| `customers.csv` | `raw.customers` | `date_of_birth`, `onboarding_date` | Ya |
| `sanctions_watchlist.csv` | `raw.sanctions_watchlist` | `listing_date` | Belum |
| `transactions.csv` | `raw.transactions` | `transaction_timestamp` | Ya; tabel fakta utama |

Import memakai `pandas.DataFrame.to_sql()` dengan:

- `schema="raw"`;
- `if_exists="replace"`;
- `method="multi"`; dan
- batch yang dibatasi agar berada di bawah limit parameter PostgreSQL.

`if_exists="replace"` berarti setiap Run All akan mengganti tabel raw target dengan isi CSV saat ini. Hal tersebut cocok untuk dataset prototype yang reproducible, tetapi jangan digunakan tanpa pertimbangan pada database produksi karena tabel lama akan diganti.

### 4.3 Primary key dan foreign key

Notebook memvalidasi ID secara logis pada tahap berikutnya, tetapi **belum membuat physical `PRIMARY KEY` atau `FOREIGN KEY` constraint di PostgreSQL**. Tabel yang dibuat `to_sql(..., if_exists="replace")` tidak otomatis membawa constraint relasional dari CSV.

Kunci logis yang dipakai di workflow ini adalah:

| Tabel | Primary key logis | Relasi penting |
|---|---|---|
| `raw.customers` | `customer_id` | Pemilik account dan sender transaksi |
| `raw.accounts` | `account_id` | `customer_id → raw.customers.customer_id` |
| `raw.counterparties` | `counterparty_id` | Penerima transaksi eksternal |
| `raw.transactions` | `transaction_id` | Sender/receiver/account/counterparty pada event |

Hal ini berarti integritas referensial masih dibuktikan lewat query dan assertion, bukan dipaksa oleh database constraint. Jika tahap prototype kelak distabilkan, constraint/index dapat ditambahkan dalam migration SQL terpisah.

## 5. Base table: join empat raw table

### 5.1 Mengapa `transactions` menjadi tabel utama

Tujuan AML monitoring adalah mengevaluasi **kejadian transaksi**, bukan sekadar menilai profil customer. Karena itu, `raw.transactions AS t` dipakai sebagai tabel utama. Semua join adalah `LEFT JOIN` agar transaksi tetap dipertahankan, termasuk transaksi external yang tidak mempunyai customer penerima internal.

Hasil yang benar harus tetap berjumlah **250.000 baris**, yaitu satu baris untuk setiap `transaction_id` pada dataset full.

### 5.2 SQL sumber

Notebook membaca file berikut, bukan menulis SQL join panjang di dalam cell Python:

```text
sql/join_four_tables.sql
```

Pemisahan ini membuat SQL dapat diperiksa, diuji, dan diubah tanpa mencampurkannya dengan logic Pandas feature engineering.

### 5.3 Join dan peran setiap tabel

| Alias SQL | Join | Peran |
|---|---|---|
| `t` | `raw.transactions` | Event utama: waktu, nominal, channel, sender, receiver, device, status |
| `sender_customer` | `t.sender_customer_id = sender_customer.customer_id` | KYC/profil customer pengirim: income, occupation, PEP, risk rating, onboarding |
| `sender_account` | `t.sender_account_id = sender_account.account_id` **dan** `t.sender_customer_id = sender_account.customer_id` | Konfirmasi rekening yang dipakai benar milik sender dan ambil account profile |
| `receiver_customer` | `t.receiver_customer_id = receiver_customer.customer_id` | Profil penerima bila penerima adalah customer internal |
| `counterparty` | `t.counterparty_id = counterparty.counterparty_id` | Profil penerima bila penerima adalah pihak eksternal |

Tabel `customers` dijoin dua kali karena satu tabel yang sama memiliki dua peran bisnis berbeda:

```text
sender_customer   = siapa yang mengirim transaksi
receiver_customer = siapa yang menerima transaksi internal
```

Ini bukan duplikasi data; ini adalah self-role join yang umum pada data transaksi.

### 5.4 Mengapa join account memakai dua kondisi

Join sender account memakai dua kondisi:

```sql
ON t.sender_account_id = sender_account.account_id
AND t.sender_customer_id = sender_account.customer_id
```

Kondisi pertama menemukan rekening. Kondisi kedua memastikan rekening tersebut memang dimiliki oleh customer yang tercatat sebagai sender pada event. Ini memberi perlindungan tambahan terhadap data yang salah relasi sebelum fitur saldo, status, atau risk level rekening digunakan.

### 5.5 Conformed receiver dengan `COALESCE`

Penerima bisa berupa customer internal atau counterparty eksternal. Jika hanya memakai kolom customer receiver, transaksi eksternal akan `NULL`; jika hanya memakai counterparty, transaksi internal akan `NULL`.

SQL membuat representasi penerima yang seragam:

| Kolom conformed | Urutan fallback |
|---|---|
| `receiver_party_id` | `receiver_customer.customer_id` → `counterparty.counterparty_id` |
| `receiver_party_name` | nama customer receiver → nama counterparty → nama pada event transaksi |
| `receiver_party_address` | alamat customer receiver → alamat counterparty → alamat pada event transaksi |
| `receiver_party_country` | negara customer receiver → negara counterparty → negara pada event transaksi |
| `receiver_party_risk_level` | risk rating customer receiver → risk level counterparty |

`COALESCE(a, b, c)` mengambil nilai pertama yang tidak `NULL` dari kiri ke kanan. Dengan demikian, satu kolom `receiver_party_*` dapat dipakai oleh feature Multiple Senders baik untuk penerima `CUS...` internal maupun `CP...` eksternal.

Kolom raw asal—`receiver_customer_id`, `receiver_account_id`, `counterparty_id`, `beneficiary_name`, dan `beneficiary_address`—tetap dipertahankan untuk traceability. Join tidak menimpa nilai source.

### 5.6 Output join

Notebook menyimpan salinan hasil join ke:

```text
notebooks/joined_tables.csv
```

File tersebut adalah artefak inspeksi/audit dari SQL join. Pembuatan feature dalam Run All memakai DataFrame `joined_tables` langsung di memory, lalu menyimpan output final Feature ABT di `data/processed`.

## 6. Standardisasi base table

### 6.1 Tujuan

Standardisasi tidak menciptakan pola AML baru. Tujuannya adalah memastikan tipe data dan kolom inti siap digunakan oleh perhitungan window, rasio, dan join pada langkah berikutnya.

Notebook terlebih dahulu memastikan delapan kolom wajib tersedia:

```text
transaction_id
transaction_timestamp
transaction_status
amount_idr_equivalent
sender_customer_id
sender_account_id
receiver_party_id
receiver_account_id
```

Jika salah satu kolom hilang, notebook menghentikan proses dengan `ValueError` sebelum menghasilkan feature yang salah.

### 6.2 Tipe data yang dipastikan

| Kolom | Transformasi | Mengapa penting |
|---|---|---|
| `transaction_timestamp` | `pd.to_datetime(..., errors="raise")` | Window 1 jam, 24 jam, 30 hari, serta selisih waktu membutuhkan datetime valid |
| `amount`, `amount_idr_equivalent`, `sender_customer_monthly_income` | `pd.to_numeric(..., errors="raise")` | Agregasi nominal dan rasio tidak boleh bekerja pada text |
| Seluruh ID inti | diubah ke Pandas `string` dan `str.strip()` | Menghindari join/flag gagal hanya karena spasi atau tipe object yang tidak konsisten |
| `transaction_status` | trim lalu title case | Memastikan `Success`, `Failed`, dan `Reversed` konsisten |

`errors="raise"` adalah pilihan yang disengaja: data kotor tidak diam-diam dipaksa menjadi `NaN`. Notebook berhenti agar sumber masalah dapat diperbaiki terlebih dahulu.

### 6.3 Lima kolom standardisasi tambahan

| Kolom | Tipe/nilai | Arti |
|---|---|---|
| `is_success` | `True` / `False` | Transaksi berstatus `Success`; hanya transaksi sukses dipakai untuk riwayat perilaku AML |
| `is_internal_receiver` | boolean | `receiver_party_id` diawali `CUS` |
| `is_external_receiver` | boolean | `receiver_party_id` diawali `CP` |
| `transaction_date` | datetime yang dinormalisasi ke 00:00 | Versi tanggal dari timestamp, berguna untuk agregasi harian/EDA |
| `transaction_hour` | integer 0–23 | Jam transaksi, berguna untuk analisis waktu tidak lazim |

Notebook meng-assert bahwa setiap baris masuk tepat ke satu kondisi penerima:

```text
internal XOR external = True
```

Artinya, transaksi tidak boleh secara bersamaan mempunyai receiver internal dan external, serta tidak boleh tidak terklasifikasi.

### 6.4 Validasi standardisasi

Sebelum membuat feature, notebook memeriksa:

- `transaction_id` unik;
- timestamp berhasil diparse;
- `amount_idr_equivalent` positif;
- `sender_customer_monthly_income` positif;
- `sender_account_id` tersedia;
- `receiver_party_id` tersedia; dan
- seluruh receiver berhasil diklasifikasi internal atau external.

Jika salah satu check gagal, feature engineering dihentikan. Hal ini mencegah kesalahan kecil—misalnya timestamp text atau ID kosong—berubah menjadi agregasi historis yang tidak dapat dipercaya.

## 7. Feature configuration

Seluruh parameter prototype dikumpulkan dalam `FEATURE_CONFIG`. Parameter ini adalah pilihan eksperimen, bukan ambang hukum/regulator.

| Area | Parameter utama | Nilai saat ini | Fungsi |
|---|---|---:|---|
| Structuring | `small_transaction_max_idr` | Rp10.000.000 | Batas nominal transaksi kecil |
| Structuring | windows aktivitas | 1h, 120m, 24h, 7d | Horizon feature transaksi sender/receiver |
| Structuring | minimum count / total 24h | 4 / Rp30.000.000 | Syarat candidate structuring |
| Rapid movement | lookback | 24 jam | Maksimum sejarah inbound yang dicari |
| Rapid movement | alert minutes | 30 menit | Jeda maksimum inbound → outbound untuk candidate |
| Rapid movement | ratio | 0,75–1,10 | Rasio outbound terhadap inbound terakhir |
| Spike | history window | 30 hari | Riwayat account untuk baseline nominal |
| Spike | minimum history / ratio prior max | 3 / 3,0× | Syarat candidate spike |
| Dormant | dormancy / minimum amount | 60 hari / Rp150.000.000 | Syarat candidate reactivation |
| Multiple senders | candidate window | 24 jam | Horizon flag candidate |
| Multiple senders | minimum sender / transaksi | 4 / 4 | Syarat candidate funnel receiver |

Parameter dipisahkan dari logika calculation agar analyst dapat melakukan kalibrasi tanpa mengganti fungsi feature. Contoh: mengubah `rapid_alert_minutes` dari 30 menjadi 60 akan mengubah candidate flag rapid tanpa mengubah cara inbound dicocokkan dengan outbound.

## 8. Prinsip leakage safety

Feature untuk transaksi pada waktu `t` hanya boleh memakai informasi yang tersedia **sebelum atau pada saat** `t`. Ia tidak boleh memakai transaksi masa depan atau outcome yang terjadi sesudahnya.

| Typology | Teknik leakage-safe |
|---|---|
| Structuring | rolling window `closed="left"` menghitung riwayat sebelum event, lalu event saat ini ditambahkan secara eksplisit |
| Rapid movement | `merge_asof(... direction="backward", allow_exact_matches=False)` hanya mencari inbound sebelum outbound |
| Spike | rolling 30 hari memakai `closed="left"`, sehingga nominal current transaction tidak masuk baseline historisnya sendiri |
| Dormant | timestamp aktivitas yang sama dikelompokkan dahulu; event pada timestamp sama tidak boleh saling menjadi history |
| Multiple senders | custom sliding window menulis feature untuk satu timestamp group sebelum group itu dimasukkan ke history |

Ground truth tidak dipakai untuk membuat satu pun feature atau candidate flag. Ground truth baru dibaca pada section evaluasi paling bawah, setelah ABT selesai dibuat.

## 9. Feature Structuring / Smurfing

### 9.1 Intuisi bisnis

Structuring terjadi ketika nominal besar dipecah menjadi banyak transaksi yang masing-masing tampak kecil. Karena itu, satu transaksi kecil tidak cukup menjadi bukti; yang dipantau adalah frekuensi serta total nominal kumpulan transaksi kecil oleh sender yang sama.

### 9.2 Populasi dan perhitungan

Hanya `successful_transactions` yang dipakai. Perhitungan dilakukan per `sender_account_id` pada empat horizon: 1 jam, 120 menit, 24 jam, dan 7 hari.

Notebook membuat dua kelompok feature:

1. **Seluruh transaksi sukses sender** (`sender_success_*`), untuk mengukur velocity umum.
2. **Transaksi sukses yang nominalnya ≤ Rp10 juta** (`sender_subthreshold_*`), untuk mengukur pola pecahan nominal kecil.

Setiap kelompok memiliki kolom berikut untuk masing-masing horizon.

| Pola nama kolom | Contoh | Arti |
|---|---|---|
| `{prefix}_txn_count_{window}` | `sender_success_txn_count_24h` | Jumlah transaksi sender dalam horizon, termasuk transaksi saat ini |
| `{prefix}_amount_sum_{window}_idr` | `sender_subthreshold_amount_sum_24h_idr` | Total IDR dalam horizon, termasuk transaksi saat ini |

Prefix yang digunakan adalah `sender_success` dan `sender_subthreshold`; window yang digunakan adalah `1h`, `120m`, `24h`, dan `7d`. Jadi, feature structuring/velocity menghasilkan 16 kolom rolling ditambah satu candidate flag.

### 9.3 Candidate flag

`is_structuring_candidate_24h = 1` bila semua syarat berikut benar:

1. transaksi saat ini sukses;
2. nominal transaksi saat ini ≤ Rp10.000.000;
3. terdapat minimal empat transaksi sub-threshold dalam 24 jam, termasuk transaksi ini; dan
4. total nominal sub-threshold dalam 24 jam minimal Rp30.000.000.

Flag tersebut adalah red flag prototype, bukan vonis AML dan bukan ketentuan regulator.

## 10. Feature Rapid Movement of Funds

### 10.1 Intuisi bisnis

Rapid movement menguji pola dana masuk ke rekening internal lalu segera keluar lagi dengan nominal yang hampir sama. Pola ini sering disebut pass-through behaviour dan dapat menjadi red flag ketika konteks customer tidak menjelaskannya.

### 10.2 Definisi inbound dan outbound

Pada data ini, semua event memiliki `debit_credit = Debit`. Karena itu, inbound tidak ditentukan dari kolom tersebut. Arah dana diturunkan dari posisi account pada event:

- **outbound event:** transaksi sukses dari `sender_account_id`;
- **internal inbound event:** transaksi sukses yang mempunyai `is_internal_receiver = True`; rekening penerimanya adalah `receiver_account_id`.

Untuk setiap outbound, `pd.merge_asof` mencari satu internal inbound **terakhir** pada account yang sama dalam 24 jam sebelumnya.

### 10.3 Kolom output

| Kolom | Nilai bila ada match | Nilai bila tidak ada match | Arti |
|---|---|---|---|
| `has_prior_internal_inbound_24h` | `1` | `0` | Ada internal inbound sebelumnya dalam lookback 24 jam |
| `minutes_since_last_internal_inbound` | Selisih menit positif | `-1` | Jeda waktu inbound terakhir ke outbound saat ini |
| `last_internal_inbound_amount_idr` | Nominal inbound terakhir | `0` | Nominal dana masuk yang cocok |
| `outbound_to_last_inbound_ratio` | outbound ÷ inbound | `0` | Perbandingan nominal keluar dan inbound terakhir |
| `is_rapid_movement_candidate` | `1` jika seluruh syarat terpenuhi | `0` | Flag red flag rapid movement |

Nilai `-1` untuk menit dan `0` untuk amount/ratio adalah sentinel feature. Bacalah bersama `has_prior_internal_inbound_24h`; `-1` tidak berarti dana masuk terjadi pada minus satu menit.

### 10.4 Candidate flag

`is_rapid_movement_candidate = 1` bila transaksi sukses memenuhi:

- mempunyai internal inbound sebelumnya;
- inbound terjadi paling lama 30 menit sebelumnya; dan
- rasio outbound terhadap inbound terakhir berada pada rentang 0,75 sampai 1,10.

## 11. Feature Sudden Transaction Spike

### 11.1 Intuisi bisnis

Satu nominal besar tidak otomatis mencurigakan. Nominal menjadi lebih relevan ketika dibandingkan dengan pola historis **rekening yang sama**. Karena itu, spike feature membandingkan transaksi saat ini dengan transaksi sukses sender dalam 30 hari sebelumnya.

### 11.2 Kolom output

| Kolom | Arti |
|---|---|
| `prior_success_txn_count_30d` | Banyak transaksi sukses sender dalam 30 hari sebelum event saat ini |
| `prior_amount_mean_30d_idr` | Rata-rata nominal historis 30 hari |
| `prior_amount_median_30d_idr` | Median nominal historis 30 hari |
| `prior_amount_max_30d_idr` | Nominal maksimum historis 30 hari |
| `amount_to_prior_median_ratio_30d` | Nominal saat ini ÷ median historis; `0` jika median belum ada |
| `amount_to_prior_max_ratio_30d` | Nominal saat ini ÷ maximum historis; `0` jika maximum belum ada |
| `has_sufficient_history_30d` | `1` bila sedikitnya ada tiga transaksi historis, selain itu `0` |
| `is_sudden_spike_candidate` | Flag candidate spike |

Jika rekening belum memiliki sejarah yang cukup, statistik historis dan ratio diisi `0`; flag `has_sufficient_history_30d` membedakan keadaan tersebut dari rasio nol yang benar-benar bermakna. `is_sudden_spike_candidate` hanya aktif jika transaksi sukses, riwayat cukup, dan nominal sekarang ≥ 3× maximum historis 30 hari.

## 12. Feature Dormant Account Reactivation

### 12.1 Intuisi bisnis

Rekening dormant tidak didefinisikan dari `opening_date` atau `account_status`. Yang diukur adalah jeda aktual antara transaksi sukses sender saat ini dengan aktivitas sukses sender sebelumnya. Pendekatan tersebut lebih relevan untuk monitoring perilaku.

### 12.2 Kolom output

| Kolom | Arti |
|---|---|
| `has_prior_successful_sender_activity` | `1` bila ada transaksi sukses sender sebelumnya |
| `days_since_prior_successful_sender_activity` | Jeda hari dari aktivitas sukses sebelumnya; `-1` bila belum memiliki riwayat |
| `is_dormant_60d` | `1` bila transaksi sukses terjadi setelah jeda minimal 60 hari |
| `is_dormant_reactivation_candidate` | `1` bila dormant 60 hari dan nominal minimal Rp150.000.000 |

Notebook terlebih dahulu menyimpan satu kombinasi unik `(sender_account_id, transaction_timestamp)`. Hal itu memastikan transaksi pada timestamp identik tidak dianggap sebagai aktivitas sebelumnya bagi satu sama lain. Nilai `-1` hanya sentinel untuk **tidak ada sejarah**, bukan durasi negatif.

## 13. Feature Multiple Senders to One Receiver

### 13.1 Intuisi bisnis

Pola ini menangkap kemungkinan funnel account: beberapa customer mengirim dana ke satu beneficiary/party dalam periode singkat. Receiver dapat berupa customer internal atau counterparty eksternal, sehingga feature memakai `receiver_party_id` conformed dari SQL join.

### 13.2 Mengapa memakai sliding window custom

Pandas rolling aggregation cukup langsung untuk count/sum sender. Namun, menghitung `nunique` sender secara rolling pada 250.000 transaksi bisa lambat. Notebook memakai `deque` dan `Counter`:

- `deque` menyimpan event yang masih berada dalam horizon;
- `Counter` menyimpan jumlah event dari setiap sender customer;
- total amount dipertahankan secara incremental.

Setiap receiver party diproses terpisah dan setiap horizon mempunyai state sendiri. Ketika event lama keluar dari horizon, count sender dan total amount dikurangi. Ini lebih hemat daripada menghitung ulang seluruh window dari awal untuk setiap baris.

### 13.3 Kolom output

Untuk masing-masing window `1h`, `120m`, `24h`, dan `7d`, notebook membuat tiga feature:

| Pola nama | Contoh | Arti |
|---|---|---|
| `receiver_txn_count_{window}` | `receiver_txn_count_24h` | Jumlah transaksi sukses ke receiver party dalam horizon |
| `distinct_senders_to_receiver_{window}` | `distinct_senders_to_receiver_24h` | Jumlah customer sender unik yang mengirim ke receiver party |
| `receiver_amount_sum_{window}_idr` | `receiver_amount_sum_24h_idr` | Total dana diterima receiver party dalam IDR |

Feature kemudian menghasilkan 12 kolom rolling dan satu flag berikut:

```text
is_multiple_senders_candidate_24h = 1
```

Flag aktif bila transaksi sukses, receiver menerima minimal empat transaksi dalam 24 jam, dan sedikitnya empat customer sender berbeda. Event dengan timestamp sama tidak dihitung sebagai sejarah satu sama lain; seluruh event pada timestamp tersebut baru dimasukkan ke window sesudah featurenya dicatat.

## 14. Ringkasan 98 kolom Feature ABT

Output final saat ini berisi **250.000 baris dan 98 kolom**. Komposisinya adalah 46 kolom hasil SQL join, lima kolom standardisasi, dan 47 kolom engineered feature.

| Kelompok | Jumlah | Isi |
|---|---:|---|
| Transaction dan context hasil join | 46 | Event, sender customer/account profile, conformed receiver, dan raw receiver traceability |
| Standardisasi | 5 | Success flag, receiver internal/external flag, tanggal, dan jam |
| Structuring/velocity | 17 | Count/sum sender sukses dan sub-threshold pada empat horizon + candidate flag |
| Rapid movement | 5 | Hubungan internal inbound terakhir dengan outbound + candidate flag |
| Spike | 8 | Baseline historis 30 hari, ratio, history flag, dan candidate flag |
| Dormant | 4 | Riwayat aktivitas sebelumnya, gap hari, dormant flag, dan candidate flag |
| Multiple senders | 13 | Receiver count/amount/distinct sender pada empat horizon + candidate flag |
| **Total** | **98** | Satu baris per transaksi |

### 14.1 Kolom candidate flag yang siap dipakai rule evaluation

| Candidate column | Typology | Rule ID pada evaluasi notebook |
|---|---|---|
| `is_structuring_candidate_24h` | Structuring / Smurfing | `RB01` |
| `is_rapid_movement_candidate` | Rapid Movement of Funds | `RB02` |
| `is_sudden_spike_candidate` | Sudden Transaction Spike | `RB03` |
| `is_dormant_reactivation_candidate` | Dormant Account Reactivation | `RB04` |
| `is_multiple_senders_candidate_24h` | Multiple Senders to One Receiver | `RB05` |

Candidate flag disimpan agar rule evaluation sederhana dapat dilakukan tanpa menghitung rolling feature ulang. Feature numerik di belakangnya tetap disimpan untuk audit, kalibrasi threshold, dan input model anomaly detection.

## 15. Save dan smoke-read Feature ABT

Setelah semua feature selesai dibuat, notebook membuat folder ini bila belum ada:

```text
data/processed/
```

Kemudian ABT ditulis ke:

```text
data/processed/transaction_feature_abt.csv
```

Cell berikutnya membaca ulang CSV dengan `pd.read_csv()` dan menampilkan `head(10)` serta daftar kolom. Tujuan read-back ini adalah smoke check sederhana: memastikan file benar-benar dapat dibuka kembali, bukan hanya DataFrame di memory.

## 16. Rule-based evaluation di akhir notebook

### 16.1 Apa yang dievaluasi

Section terakhir hanya menggunakan lima candidate flag yang sudah ada di ABT. Ia membuat katalog rule sederhana:

| Rule | Candidate flag | Scenario ground truth pembanding |
|---|---|---|
| `RB01` Structuring / Smurfing | `is_structuring_candidate_24h` | `AML-S01` |
| `RB02` Rapid Movement of Funds | `is_rapid_movement_candidate` | `AML-S03` |
| `RB03` Sudden Transaction Spike | `is_sudden_spike_candidate` | `AML-S02` |
| `RB04` Dormant Account Reactivation | `is_dormant_reactivation_candidate` | `AML-S04` |
| `RB05` Multiple Senders to One Receiver | `is_multiple_senders_candidate_24h` | `AML-S05` |

Rule section membaca hanya `transaction_id` dan lima flag dari ABT supaya ringan. Ground truth dibaca terpisah dari `data/ground_truth/aml_ground_truth.csv` dan baru dijoin pada evaluasi.

### 16.2 Metrik yang dihitung

| Metrik | Rumus | Arti |
|---|---|---|
| `hit_count` | jumlah baris dengan flag = 1 | Volume alert dari rule |
| `hit_rate_pct` | `hit_count / seluruh transaksi × 100` | Proporsi populasi yang dialertkan |
| `ground_truth_hits` | ground truth row dengan flag = 1 | Jumlah transaksi synthetic berlabel yang terdeteksi |
| `recall_pct` | `ground_truth_hits / ground_truth_rows × 100` | Sensitivitas rule terhadap scenario synthetic yang sesuai |

Recall menggunakan unit transaksi (*row-level recall*). Untuk pola multi-transaksi seperti Structuring dan Multiple Senders, transaksi awal dalam group dapat belum terflag karena count atau amount threshold belum tercapai. Karena itu, recall per transaksi kurang dari 100% tidak otomatis berarti bug. Itu dapat menjadi konsekuensi normal dari trailing-window monitoring real-time.

### 16.3 Batas evaluasi ini

Evaluasi saat ini belum menghitung precision, false-positive rate, case-level recall, atau workload investigator. Selain itu, ia tidak mengevaluasi lima scenario tambahan (`AML-S06` sampai `AML-S10`) karena belum terdapat candidate flag terkait di ABT. Hasil evaluasi harus dibaca sebagai validasi prototype data dan threshold, bukan estimasi performa AML production.

## 17. Ground truth dan ML: aturan anti-leakage

`aml_ground_truth.csv` berisi jawaban synthetic, seperti `scenario_id`, `injected_flag`, dan `expected_risk`. Kolom tersebut tidak boleh masuk ke Feature ABT atau training model unsupervised.

Alur yang benar:

```text
Raw data + feature ABT  -->  rule/model membuat score atau flag
                                      |
                                      v
                       hasil detection di-join ke ground truth
                                      |
                                      v
                          recall / precision / ranking evaluation
```

Alur yang salah adalah memasukkan `scenario_id` atau `injected_flag` ketika menghitung feature atau melatih Isolation Forest/LOF. Itu akan membuat metrik evaluasi tampak sangat tinggi karena model sudah mengetahui jawaban.

## 18. Troubleshooting

| Gejala | Kemungkinan penyebab | Tindakan |
|---|---|---|
| Password PostgreSQL ditolak | Password/role/port salah atau server tidak berjalan | Cek service PostgreSQL, `POSTGRES_USER`, port 5432, lalu ulangi setup |
| `Folder raw tidak ditemukan` | Notebook dijalankan dari lokasi yang salah atau Notebook 01 belum dijalankan | Buka dari project `notebooks` dan pastikan `data/raw` ada |
| `sql/join_four_tables.sql` tidak ditemukan | `PROJECT_ROOT` hard-coded tidak cocok dengan lokasi project | Ubah `PROJECT_ROOT` atau kembalikan struktur folder project |
| Hasil join bukan 250.000 baris | SQL diubah menjadi inner join atau key tidak unik | Pastikan basisnya `raw.transactions`, seluruh join `LEFT JOIN`, dan cek duplicate master key |
| Assertion receiver classification gagal | `receiver_party_id` tidak menghasilkan ID `CUS...` atau `CP...` | Periksa `COALESCE` pada SQL join serta data source receiver/counterparty |
| Feature rolling tampak nol pada transaksi gagal | Ini normal | Historical feature aktif dihitung untuk transaksi sukses; transaksi non-success memakai default feature value |
| Error `NameError: transaction_feature_abt` | Cell feature dijalankan sebelum standardisasi/configuration | Jalankan dari setup sampai section tersebut secara berurutan atau Run All |
| Candidate flag terlalu banyak/sedikit | Parameter threshold prototype belum terkalibrasi | Ubah nilai di `FEATURE_CONFIG`, rebuild ABT, lalu evaluasi ulang terhadap ground truth |

## 19. Checklist handoff

- [ ] Semua lima raw CSV berhasil dimuat ke schema PostgreSQL `raw`.
- [ ] `joined_tables` tetap satu baris per `transaction_id` dan berjumlah 250.000 untuk data full.
- [ ] Seluruh `standardisation_checks` bernilai `True`.
- [ ] `transaction_feature_abt.csv` berhasil dibaca ulang sesudah disimpan.
- [ ] Ground truth belum dipakai sebagai input feature ataupun model.
- [ ] Lima candidate flag dapat dievaluasi terhadap scenario `AML-S01`–`AML-S05`.
- [ ] ABT siap digunakan sebagai input rule-based monitoring atau feature source untuk notebook ML anomaly detection.
