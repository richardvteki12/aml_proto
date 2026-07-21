# 01 — Generate Synthetic AML Data

Dokumen ini menjelaskan cara kerja notebook [`01_generate_synthetic_data.ipynb`](../notebooks/01_generate_synthetic_data.ipynb). Notebook tersebut adalah titik awal pipeline: ia membuat data AML **sepenuhnya sintetis**, menyimpannya ke CSV, dan menghasilkan ground truth untuk pengujian tahap berikutnya.

> Status scope: notebook ini hanya membuat data. Notebook ini **belum** membangun Analytical Base Table (ABT), feature engineering, rule-based alert, model machine learning, maupun dashboard.

## 1. Tujuan notebook

Notebook membuat dataset yang bentuk dan hubungannya menyerupai lingkungan perbankan, tetapi tidak memuat data nasabah nyata. Dataset digunakan untuk:

1. mempelajari struktur data AML;
2. melakukan EDA pada `02_eda.ipynb`;
3. membangun join dan feature engineering pada notebook berikutnya;
4. menguji apakah feature atau rule dapat menemukan pola AML yang sengaja diinjeksi; dan
5. mengevaluasi model anomaly detection tanpa menggunakan label ketika training.

Semua ambang (threshold), nominal, jumlah baris, serta perilaku AML di sini adalah **parameter prototype synthetic**, bukan ketentuan hukum atau ambang pelaporan resmi dari regulator. Dalam implementasi sungguhan, parameter harus dikalibrasi lewat Risk-Based Approach (RBA), profil nasabah, kebijakan internal, serta tata kelola AML lembaga.

## 2. Alur besar proses

```text
Konfigurasi + random seed
          |
          v
Master data: customers, accounts, counterparties, watchlist
          |
          v
Baseline transaksi normal (250.000 transaksi pada mode full)
          |
          v
Injeksi 10 skenario AML + variasi nama sanctions
          |
          v
Perbaikan relasi party dan standardisasi nilai kosong
          |
          v
Acceptance checks / validasi referential integrity
          |
          v
CSV raw  +  CSV ground truth
```

Urutan tersebut penting. Master data harus ada sebelum transaksi dibuat karena transaksi membutuhkan ID pengirim, rekening pengirim, serta pihak penerima. Injeksi AML dilakukan setelah baseline agar pola mencurigakan hadir di tengah populasi transaksi normal. Terakhir, notebook memvalidasi bahwa perubahan injeksi tidak merusak relasi data.

## 3. Lokasi input dan output

Notebook otomatis mencari project root dengan naik dari folder `notebooks` sampai menemukan folder tersebut. Dengan demikian, notebook tidak mengandalkan path absolut komputer tertentu.

Output yang dibuat berada di bawah project root berikut.

```text
data/
├── raw/
│   ├── customers.csv
│   ├── accounts.csv
│   ├── counterparties.csv
│   ├── sanctions_watchlist.csv
│   └── transactions.csv
└── ground_truth/
    ├── aml_ground_truth.csv
    └── sanctions_ground_truth.csv
```

`data/raw` adalah data operasional sintetis yang boleh dipakai sebagai input EDA, SQL, dbt, ABT, rule, atau model. `data/ground_truth` adalah data jawaban referensi yang **tidak boleh menjadi feature atau input training unsupervised**. Ground truth hanya dipakai setelah scoring untuk evaluasi dan pengujian.

## 4. Cara menjalankan

### Prasyarat

Kernel Python notebook harus mempunyai minimal paket berikut.

```python
import numpy as np
import pandas as pd
```

Notebook juga mengimpor fungsi dari:

```text
scripts/aml_scenario_injection.py
```

Jadi folder `scripts` harus tetap berada di project root dan notebook perlu dijalankan dari struktur folder project yang utuh.

### Urutan eksekusi

1. Buka `notebooks/01_generate_synthetic_data.ipynb`.
2. Pada bagian parameter, pilih `RUN_SCALE`:
   - `"smoke"` untuk cek cepat selama pengembangan;
   - `"full"` untuk dataset final.
3. Biarkan `SEED = 42` bila ingin hasil yang konsisten dan reproducible.
4. Klik **Run All**.
5. Pastikan seluruh `assert` pada bagian validasi selesai tanpa error.
6. Periksa tabel daftar file pada cell terakhir dan pastikan tujuh CSV telah tertulis.

Menjalankan ulang notebook akan menulis ulang CSV output dengan hasil yang sama selama kode, `SEED`, dan parameter tidak berubah. Jangan mengubah nilai seed di tengah eksperimen bila hasil sebelumnya masih ingin dibandingkan secara langsung.

## 5. Konfigurasi ukuran data

Cell parameter memakai dua skala data berikut.

| Parameter | `full` | `smoke` | Arti |
|---|---:|---:|---|
| `customers` | 10.000 | 500 | Jumlah master nasabah internal |
| `accounts` | 15.000 | 750 | Jumlah rekening internal |
| `counterparties` | 5.000 | 300 | Jumlah pihak eksternal |
| `watchlist` | 1.200 | 120 | Jumlah entitas pada watchlist sanctions sintetis |
| `transactions` | 250.000 | 6.000 | Jumlah transaksi baseline sebelum/selama injeksi |
| `history_days` | 240 hari | 60 hari | Panjang riwayat transaksi |
| `labels_per_scenario` | 100 | 20 | Jumlah transaksi berlabel untuk setiap skenario AML |
| `sanctions_cases` | 750 | 60 | Jumlah kasus kandidat sanctions yang diinjeksi |

Untuk dataset final, gunakan `RUN_SCALE = "full"`. Mode `smoke` sengaja lebih kecil supaya perubahan kode dapat diuji cepat; ia bukan pengganti data final untuk evaluasi proyek.

## 6. Pembentukan master data

### 6.1 `customers.csv` — satu baris per nasabah internal

**Grain:** tepat satu record untuk satu `customer_id`.

Tabel ini menyimpan profil KYC sintetis. Kunci utamanya adalah `customer_id` dengan format seperti `CUS0000001`. Data tersebut dipakai untuk menghubungkan pemilik rekening dan profil pengirim/penerima internal.

| Kelompok kolom | Kolom | Kegunaan |
|---|---|---|
| Identitas | `customer_id`, `full_name`, `first_name`, `middle_name`, `last_name`, `date_of_birth`, `gender`, `nationality`, `id_type`, `id_number` | Identifikasi dan CDD dasar sintetis |
| Alamat dan kontak | `address_line_1`, `address_line_2`, `city`, `province`, `postal_code`, `country`, `phone_number`, `email` | Profil lokasi dan kontak |
| Ekonomi dan pekerjaan | `occupation`, `employer_name`, `monthly_income`, `customer_segment` | Konteks kemampuan dan profil transaksi nasabah |
| Risiko dan lifecycle | `customer_risk_rating`, `pep_flag`, `onboarding_date`, `account_status` | Input KYC/risk-based monitoring |

`monthly_income` dan saldo pada tabel rekening dinyatakan dalam **Rupiah (IDR)**, bukan jutaan atau miliaran sebagai unit terpisah. Contoh `12_000_000` berarti Rp12.000.000.

### 6.2 `accounts.csv` — satu baris per rekening internal

**Grain:** tepat satu record untuk satu `account_id`.

| Kelompok kolom | Kolom | Kegunaan |
|---|---|---|
| Identitas rekening | `account_id`, `account_number`, `account_type`, `currency`, `branch_code` | Identitas serta karakteristik produk rekening |
| Relasi pemilik | `customer_id` | Foreign key ke `customers.customer_id` |
| Lifecycle dan risiko | `opening_date`, `account_status`, `risk_level` | Usia, status, serta risk tier rekening |
| Nilai finansial | `average_balance`, `current_balance` | Konteks nominal transaksi terhadap rekening |

Satu customer dapat memiliki lebih dari satu rekening. Generator lebih dulu memberi setiap customer minimal satu rekening, lalu membagikan sisa rekening secara acak. Karena itu, relasi customer → account adalah **one-to-many**.

### 6.3 `counterparties.csv` — satu baris per pihak eksternal

**Grain:** tepat satu record untuk satu `counterparty_id`.

Counterparty bukan customer bank internal. Tabel ini dipakai bila transaksi keluar menuju pihak/akun eksternal.

| Kelompok kolom | Kolom |
|---|---|
| Identitas | `counterparty_id`, `counterparty_name`, `entity_type`, `alias_name` |
| Identitas individu/perusahaan | `date_of_birth`, `registration_date`, `nationality` |
| Lokasi dan bank | `address`, `city`, `country`, `bank_name`, `account_number` |
| Risiko | `industry`, `risk_level` |
| Indikator applicability | `has_date_of_birth`, `has_registration_date` |

`entity_type` bernilai `Individual` atau `Company`. Tanggal lahir hanya relevan untuk individu; tanggal registrasi hanya relevan untuk perusahaan. Agar hasil ekspor tidak memiliki `NaN`, kolom yang tidak berlaku memakai tanggal sentinel `1900-01-01` dan indikator `has_date_of_birth` / `has_registration_date` menjelaskan apakah tanggal itu memang berlaku. Pada tahap analisis, gunakan indikator tersebut; jangan memperlakukan tanggal sentinel sebagai tanggal asli.

### 6.4 `sanctions_watchlist.csv` — satu baris per entitas watchlist

**Grain:** tepat satu record untuk satu `watchlist_id`.

Tabel ini adalah daftar referensi untuk eksperimen name screening. Isinya bukan daftar sanctions sungguhan. Field pentingnya mencakup:

| Kelompok | Kolom |
|---|---|
| Identitas watchlist | `watchlist_id`, `entity_type`, `primary_name`, `aliases` |
| Atribut identitas | `first_name`, `middle_name`, `last_name`, `date_of_birth`, `place_of_birth`, `nationality`, `passport_number`, `national_id`, `organization_name` |
| Lokasi dan listing | `address`, `city`, `country`, `program`, `listing_reason`, `list_source`, `listing_date`, `active_flag`, `risk_level` |
| Applicability | `has_date_of_birth` |

Untuk organisasi, `first_name`, `last_name`, dan tanggal lahir tidak bermakna. Untuk individu, `organization_name` tidak bermakna. Notebook menggunakan nilai sentinel tekstual yang eksplisit, bukan string kosong, sehingga pembacaan CSV tidak otomatis mengubah nilai menjadi `NaN`.

## 7. Pembentukan baseline `transactions.csv`

### 7.1 Grain dan periode waktu

**Grain:** tepat satu baris untuk satu `transaction_id`.

Pada mode `full`, generator membentuk 250.000 transaksi pada periode 240 hari hingga **2026-06-30 23:59:59**. Timestamp dibuat acak dalam rentang tersebut. `amount_idr_equivalent` selalu dinyatakan dalam Rupiah dan merupakan kolom nominal utama untuk perbandingan lintas currency.

Mata uang transaksi dibangkitkan dari campuran IDR, USD, SGD, dan EUR. Kolom `amount` adalah nominal pada currency asal, sedangkan `amount_idr_equivalent` adalah nilai yang telah dikonversi menggunakan kurs synthetic yang disimpan di notebook. Oleh sebab itu, untuk analisis nominal AML lintas mata uang, gunakan `amount_idr_equivalent`.

### 7.2 Kolom transaksi

| Kelompok | Kolom | Penjelasan |
|---|---|---|
| Kunci dan waktu | `transaction_id`, `transaction_timestamp` | Identitas unik dan waktu kejadian |
| Pengirim internal | `sender_customer_id`, `sender_account_id`, `sender_name`, `sender_address`, `sender_country` | Pengirim selalu direlasikan ke customer dan account internal |
| Penerima / beneficiary | `receiver_customer_id`, `receiver_account_id`, `receiver_name`, `receiver_address`, `receiver_country`, `beneficiary_name`, `beneficiary_address`, `counterparty_id` | Menjelaskan apakah penerima internal atau eksternal serta identitasnya |
| Detail transaksi | `transaction_type`, `channel`, `amount`, `currency`, `amount_idr_equivalent`, `debit_credit`, `purpose_code`, `purpose_description`, `reference_number`, `source_of_fund` | Karakteristik finansial dan narasi transaksi |
| Tujuan dan perangkat | `destination_bank`, `destination_country`, `ip_address`, `device_id`, `latitude`, `longitude` | Koridor pembayaran dan konteks channel/device |
| Hasil proses | `transaction_status` | Status `Success`, `Failed`, atau `Reversed` |

Transaction type yang dibangkitkan adalah `Transfer`, `Cash`, `RTGS`, `SWIFT`, dan `BI-FAST`. Channel dibangkitkan dari `Mobile`, `Internet`, `Branch`, `ATM`, atau `API`.

### 7.3 Relasi internal vs eksternal

Sekitar 28% baseline transaksi dibuat sebagai transfer internal (on-us). Sisanya menggunakan external counterparty. Relasi tersebut direpresentasikan secara eksplisit sebagai berikut.

| Kondisi penerima | `receiver_customer_id` | `receiver_account_id` | `counterparty_id` |
|---|---|---|---|
| Penerima internal | ID customer internal (`CUS...`) | ID account internal (`ACC...`) | `INTERNAL_ON_US_TRANSFER` |
| Penerima eksternal | `EXTERNAL_NOT_BANK_CUSTOMER` | `EXTERNAL_ACCOUNT_NOT_ON_US` | ID counterparty eksternal (`CP...`) |

Tiga teks kapital tersebut adalah **sentinel relationship values**, bukan ID customer, account, atau counterparty sebenarnya. Sentinel digunakan agar CSV tidak menghasilkan `NaN` ketika field tersebut secara struktur tidak berlaku. Saat melakukan join atau feature engineering:

- lakukan join ke `customers` / `accounts` hanya ketika receiver memakai ID `CUS...` / `ACC...` yang valid;
- lakukan join ke `counterparties` hanya ketika `counterparty_id` adalah ID `CP...` yang valid;
- jangan mencoba mencari customer atau account bernama `EXTERNAL_NOT_BANK_CUSTOMER` atau `EXTERNAL_ACCOUNT_NOT_ON_US`.

## 8. Injeksi skenario AML

Setelah baseline dibuat, notebook memanggil `scripts/aml_scenario_injection.py`. Modul ini memilih baris transaksi yang berbeda dan mengubahnya menjadi pola AML sintetis yang saling terhubung. Baris untuk scenario AML di-reserve supaya satu transaksi tidak diberi dua label AML yang berbeda. Kasus sanctions juga dipilih dari indeks yang tidak di-reserve, sehingga kasus sanctions tidak tumpang tindih dengan transaksi AML berlabel.

Pada mode `full`, tiap scenario memiliki 100 transaksi berlabel. Itu menghasilkan **1.000 baris** pada `aml_ground_truth.csv`: 100 transaksi × 10 scenario. Tidak semua scenario mempunyai jumlah *group* yang sama, karena beberapa pola membutuhkan empat transaksi sebagai satu bukti perilaku.

### 8.1 Lima scenario fokus untuk feature engineering berikutnya

| ID | Typology | Cara injeksi | Bukti yang nantinya dapat dipakai feature/rule | Risiko |
|---|---|---|---|---|
| `AML-S01` | Structuring / Smurfing | Empat transfer outbound oleh satu rekening menuju dua counterparty eksternal dalam maksimal 120 menit. Setiap transaksi berada pada 91%–98,5% dari Rp10 juta, tetapi total grup minimal Rp30 juta. | Jumlah transaksi, total nominal, maksimum nominal, dan distinct counterparty dalam trailing window | High |
| `AML-S02` | Sudden Transaction Spike | Transaksi sukses dipilih hanya jika rekening memiliki riwayat sukses 30 hari yang cukup. Nominal baru dibuat minimal 3× maximum historis sebelumnya, 8× median historis sebelumnya, atau Rp150 juta. | Prior count, prior median/max, rasio current amount terhadap baseline sebelumnya | High |
| `AML-S03` | Rapid Movement of Funds | Dana masuk internal ke rekening target, lalu 5–20 menit kemudian keluar ke counterparty eksternal. Nominal outbound adalah 80%–105% inbound dan harus masih berada dalam jendela 24 jam. Hanya leg outbound yang diberi label AML. | Inbound sebelumnya ke rekening yang sama, selisih waktu inbound→outbound, rasio outbound/inbound | Critical |
| `AML-S04` | Dormant Account Reactivation | Dipilih rekening yang mempunyai jeda aktivitas sukses sebelumnya minimal 60 hari pada mode full; lalu dibuat transfer outbound besar Rp180 juta–Rp1,1 miliar. | Waktu sejak transaksi sukses sebelumnya dan nominal transaksi saat reaktivasi | Critical |
| `AML-S05` | Multiple Senders to One Receiver | Empat customer internal berbeda mengirim ke satu counterparty eksternal dalam maksimal 120 menit. | Distinct sender customer, jumlah transaksi, dan receiver party yang sama pada trailing window | High |

Semua sejarah pada scenario spike dan dormant diperiksa secara **strictly prior**: transaksi pada waktu yang sama atau waktu setelah transaksi target tidak boleh menjadi sejarah. Ini penting agar feature pada tahap berikutnya tidak mengalami data leakage.

### 8.2 Lima scenario tambahan dalam katalog ground truth

| ID | Typology | Bentuk injeksi ringkas | Risiko |
|---|---|---|---|
| `AML-S06` | One Sender to Multiple Beneficiaries | Satu rekening mengirim ke empat counterparty eksternal berbeda dalam satu group | High |
| `AML-S07` | Circular Transaction | Relasi sender dan receiver dibuat internal (on-us) untuk scenario yang dipertahankan dalam scope 10 scenario | Critical |
| `AML-S08` | High-Risk Geography | Penerima eksternal diarahkan ke negara synthetic berisiko tinggi: `RU`, `SY`, atau `KP` | High |
| `AML-S09` | Unusual Transaction Purpose | Purpose diubah menjadi pengadaan mesin industri dengan nominal Rp350 juta–Rp900 juta | High |
| `AML-S10` | Potential Mule Account | Purpose diubah menjadi synthetic mule-account movement dengan nominal Rp35 juta–Rp180 juta | Critical |

Lima scenario tambahan tetap ada untuk mempertahankan katalog 10 typology dari assignment awal. Namun, acceptance check perilaku yang paling rinci di notebook saat ini berfokus pada `AML-S01` sampai `AML-S05`.

## 9. Ground truth

### 9.1 `aml_ground_truth.csv`

Tabel ini adalah jawaban referensi untuk scenario AML yang diinjeksi.

| Kolom | Arti |
|---|---|
| `transaction_id` | Transaksi yang dibuat suspicious oleh injeksi scenario |
| `customer_id` | Customer pengirim dari transaksi berlabel |
| `scenario_id` | ID typology, misalnya `AML-S03` |
| `scenario_name` | Nama typology yang mudah dibaca |
| `injected_flag` | Bernilai `1` untuk transaksi hasil injeksi |
| `expected_risk` | Severity prototype yang diharapkan |
| `notes` | Deskripsi singkat perilaku yang diinjeksi |
| `scenario_group_id` | Mengikat transaksi yang merupakan bagian dari pola yang sama |

Gunakan tabel ini setelah hasil rule atau model tersedia. Contoh penggunaan yang benar:

1. jalankan rule atau model hanya dengan data dari `data/raw`;
2. hasilkan `alert_flag`, `anomaly_score`, atau ranking transaksi;
3. join hasil tersebut ke `aml_ground_truth` menggunakan `transaction_id`;
4. hitung recall, precision, hit rate, dan breakdown per `scenario_id`.

Jangan memasukkan `scenario_id`, `injected_flag`, `expected_risk`, `notes`, atau `scenario_group_id` sebagai feature training; semua kolom tersebut akan membocorkan jawaban.

### 9.2 `sanctions_ground_truth.csv`

Tabel ini berbeda dari `sanctions_watchlist.csv`.

- `sanctions_watchlist.csv` adalah **daftar referensi** entitas yang perlu dicocokkan oleh proses screening.
- `sanctions_ground_truth.csv` adalah **jawaban pengujian**: transaksi mana yang telah diubah agar namanya terkait dengan sebuah entitas watchlist dan bentuk variasi nama apa yang digunakan.

| Kolom | Arti |
|---|---|
| `transaction_id` | Transaksi yang receiver/beneficiary-nya dimodifikasi |
| `watchlist_id` | Entitas referensi pada `sanctions_watchlist` |
| `injected_name` | Nama yang ditaruh ke data transaksi |
| `original_name` | Nama utama pada watchlist |
| `variation_type` | `Exact`, `Alias`, `Reversed`, `Abbreviation`, `Spacing`, atau `Typo` |
| `expected_match` | Bernilai `1`: screening diharapkan menemukan kandidat match |

Tujuannya adalah menguji name matching yang tidak hanya bergantung pada exact match, misalnya nama dengan urutan dibalik, singkatan, alias, spasi ganda, atau typo.

## 10. Standardisasi nilai kosong dan sentinel

Setelah scenario injection, notebook menyelaraskan ulang atribut sender/receiver ke master account dan customer. Langkah ini memastikan `sender_account_id → sender_customer_id` tetap valid dan penerima internal kembali sesuai dengan owner rekeningnya.

Notebook kemudian memastikan **tidak ada `NaN` dan tidak ada string kosong** pada tujuh tabel yang diekspor. Ini dilakukan untuk membuat tahap impor CSV, PostgreSQL, dan pemodelan lebih stabil. Namun, nilai sentinel harus dipahami sebagai *not applicable*, bukan fakta bisnis.

| Tabel | Field | Nilai ketika tidak berlaku | Cara membacanya |
|---|---|---|---|
| `customers` | `middle_name` | `NO_MIDDLE_NAME` | Customer memang tidak mempunyai middle name |
| `customers` | `address_line_2` | `NO_ADDRESS_LINE_2` | Baris alamat kedua tidak disediakan |
| `counterparties` | `date_of_birth` / `registration_date` | `1900-01-01` | Lihat `has_date_of_birth` / `has_registration_date` sebelum memakai tanggal |
| `sanctions_watchlist` | name fields atau organization field | `NOT_APPLICABLE_INDIVIDUAL` / `NOT_APPLICABLE_ORGANIZATION` | Field tidak relevan bagi tipe entitas tersebut |
| `transactions` | penerima customer eksternal | `EXTERNAL_NOT_BANK_CUSTOMER` | Bukan customer bank internal |
| `transactions` | rekening penerima eksternal | `EXTERNAL_ACCOUNT_NOT_ON_US` | Bukan rekening internal/on-us |
| `transactions` | counterparty untuk transfer internal | `INTERNAL_ON_US_TRANSFER` | Transaksi internal, tidak ada counterparty eksternal |

Jika kelak ingin mempertahankan structural missing value sebagai `NULL` di data warehouse, dapat dilakukan pada layer staging dengan mapping sentinel → `NULL` serta indicator flag. Jangan menghapus sentinel di raw CSV tanpa menyiapkan strategi tersebut karena join dan audit relasi dapat menjadi ambigu.

## 11. Validasi yang dijalankan sebelum CSV ditulis

Cell terakhir menjalankan acceptance checks. Notebook berhenti dengan `AssertionError` bila salah satu syarat berikut gagal.

### 11.1 Integritas data umum

- `customer_id`, `account_id`, dan `transaction_id` harus unik pada tabel masing-masing.
- Setiap `accounts.customer_id` harus tersedia di `customers.customer_id`.
- Setiap `transactions.sender_customer_id` harus tersedia di `customers.customer_id`.
- Semua `amount_idr_equivalent` harus lebih besar dari nol.
- Tidak boleh tersisa cell `NaN` atau text kosong pada tabel ekspor.

### 11.2 Traceability ground truth

- Harus terdapat tepat 10 `scenario_id` AML.
- Semua `aml_ground_truth.transaction_id` harus ditemukan di `transactions.csv`.
- `customer_id` pada AML ground truth harus cocok dengan sender customer dari transaksi yang sama.
- Semua `sanctions_ground_truth.transaction_id` harus ditemukan di `transactions.csv`.

### 11.3 Validasi perilaku lima scenario fokus

- **S01:** empat transaksi per group, satu sender account, berada dalam jendela 120 menit, tiap transaksi di bawah threshold prototype, total group melampaui threshold prototype.
- **S02:** transaksi spike memiliki riwayat 30 hari yang cukup dan melampaui maximum historis sebelumnya dengan kelipatan yang dikonfigurasi.
- **S03:** outbound berlabel benar-benar mengikuti inbound internal yang cocok, dalam jendela waktu, dan mempunyai rasio nominal yang tepat.
- **S04:** reaktivasi benar-benar mempunyai jeda aktivitas sebelumnya sesuai parameter dormancy.
- **S05:** group berisi empat transaksi, customer pengirim yang cukup berbeda, satu receiver party, dan semua transaksi berada dalam jendela waktu.

Validasi ini menguji desain synthetic data, bukan membuktikan bahwa rule atau model telah berhasil mendeteksi pattern. Deteksi dan metrik evaluasi baru dilakukan pada notebook feature engineering/rule/model.

## 12. Hubungan antar tabel untuk tahap berikutnya

```text
customers.customer_id
        1
        |
        | owns
        v
accounts.customer_id  -----> accounts.account_id
                                  |
                                  | sends
                                  v
transactions.sender_account_id

transactions.receiver_account_id  -----> accounts.account_id
    hanya jika receiver internal

transactions.counterparty_id  -----> counterparties.counterparty_id
    hanya jika receiver external

aml_ground_truth.transaction_id -----> transactions.transaction_id
sanctions_ground_truth.transaction_id -> transactions.transaction_id
sanctions_ground_truth.watchlist_id --> sanctions_watchlist.watchlist_id
```

Untuk join yang aman pada SQL atau Pandas:

1. mulai dari `transactions` sebagai tabel fakta utama;
2. join sender account ke `accounts`, lalu ke `customers` untuk memahami customer pengirim;
3. lakukan join receiver account/customer hanya untuk penerima internal;
4. lakukan join counterparty hanya untuk penerima eksternal;
5. simpan `aml_ground_truth` dan `sanctions_ground_truth` terpisah dari feature table sampai fase evaluasi.

## 13. Batasan dan prinsip penggunaan

- Dataset ini sintetis. Nama, nomor identitas, alamat, dan transaksi tidak merepresentasikan individu atau kejadian nyata.
- Injeksi AML menciptakan **ground truth desain**, bukan putusan bahwa pola yang serupa di data nyata pasti merupakan TPPU.
- Nilai `expected_risk` adalah severity prototype; nilai tersebut bukan klasifikasi regulator.
- Beberapa scenario sengaja dibuat jelas agar rule/feature dapat diuji. Di dunia nyata sinyal dapat lebih berisik, saling tumpang tindih, dan memerlukan investigasi manusia.
- Ground truth berfungsi untuk evaluasi offline. Pada anomaly detection unsupervised, label tidak digunakan dalam training.

## 14. Troubleshooting singkat

| Gejala | Kemungkinan penyebab | Tindakan |
|---|---|---|
| `ModuleNotFoundError: scripts...` | Notebook tidak menemukan project root atau folder `scripts` hilang | Pastikan notebook tetap di folder `notebooks` dan `scripts/aml_scenario_injection.py` ada di project root |
| Assertion validasi gagal | Ada perubahan generator yang merusak relasi atau perilaku scenario | Baca tabel check yang ditampilkan, kembalikan parameter/logic terkait, lalu Run All ulang |
| Jumlah file/row tidak sesuai | `RUN_SCALE` memakai `smoke` atau output lama sedang terbuka | Cek parameter lalu tutup file CSV yang sedang dipakai aplikasi lain dan jalankan ulang |
| Hasil berubah antar-run | `SEED`, parameter, atau kode generator berubah | Tetapkan `SEED = 42` dan catat versi parameter yang dipakai |
| Nilai sentinel dianggap data asli | Mapping sentinel belum diterapkan di query | Terapkan kondisi internal/eksternal seperti pada tabel relasi di atas sebelum join atau agregasi |

## 15. Checklist handoff ke notebook berikutnya

- [ ] Notebook 01 selesai tanpa error.
- [ ] Semua seven CSV berada di `data/raw` dan `data/ground_truth`.
- [ ] Tabel validasi menunjukkan seluruh check bernilai `True`.
- [ ] `RUN_SCALE` dan `SEED` dicatat.
- [ ] Ground truth disimpan terpisah dari input feature/model.
- [ ] EDA dimulai dari `02_eda.ipynb` menggunakan file pada `data/raw`.
