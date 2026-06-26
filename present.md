# Presentasi Proyek Prediksi Minat Pembelian E-Commerce

## Pendahuluan

Proyek ini bertujuan untuk memprediksi apakah pengunjung website e-commerce akan melakukan pembelian atau tidak berdasarkan perilaku kunjungan mereka. Proyek ini menggunakan teknik Machine Learning - Klasifikasi, di mana target prediksi berupa kelas diskrit: **0 = Tidak Membeli** dan **1 = Membeli**.

Dataset yang digunakan adalah `online_shoppers_intention.csv` dengan target prediksi pada kolom `Revenue`.

---

## Bagian 1: File `prediksi_pembelian_ecommerce.py`

### 1.1 Import Library

Pada tahap ini dilakukan pemanggilan library yang dibutuhkan untuk berbagai keperluan:

- **Pandas (pd)**: Untuk membaca dan memanipulasi dataset dalam bentuk DataFrame
- **NumPy (np)**: Untuk operasi array dan komputasi numerik
- **Matplotlib (plt)**: Untuk membuat visualisasi grafik dasar
- **Seaborn (sns)**: Untuk membuat visualisasi statistik yang lebih menarik
- **Joblib**: Untuk menyimpan dan memuat model machine learning
- **Scikit-learn**: Library utama untuk machine learning yang mencakup:
  - `train_test_split`: Memisahkan data menjadi training dan testing
  - `StandardScaler`: Menstandarkan skala fitur numerik
  - `OneHotEncoder`: Mengubah fitur kategorikal menjadi numerik
  - `ColumnTransformer`: Menggabungkan preprocessing berbeda dalam satu pipeline
  - `Pipeline`: Membuat alur preprocessing dan modeling
- **Model Klasifikasi**:
  - `LogisticRegression`: Regresi logistik untuk klasifikasi biner
  - `DecisionTreeClassifier`: Pohon keputusan
  - `RandomForestClassifier`: Random Forest dengan banyak pohon keputusan
  - `GradientBoostingClassifier`: Gradient Boosting
  - `XGBClassifier`: XGBoost, algoritma boosting yang sangat populer
  - `KNeighborsClassifier`: K-Nearest Neighbors
  - `SVC`: Support Vector Classifier
- **Imblearn SMOTE**: Untuk menangani ketidakseimbangan kelas dengan membuat data sintetis
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC

### 1.2 Load Dataset

Dataset dibaca menggunakan `pd.read_csv()` dari file `data/online_shoppers_intention.csv`. Dataset ini berisi data perilaku pengunjung website e-commerce. Kolom `Revenue` digunakan sebagai target karena menunjukkan apakah pengunjung melakukan pembelian atau tidak.

### 1.3 Data Understanding

Tahap ini dilakukan untuk memahami struktur dataset:

- **df.info()**: Menampilkan informasi tipe data setiap kolom, jumlah non-null values
- **df.describe()**: Menampilkan statistik deskriptif (mean, std, min, max, quartile)
- **df.columns.tolist()**: Menampilkan daftar semua nama kolom

### 1.4 Exploratory Data Analysis (EDA)

Tahap EDA dilakukan untuk memahami karakteristik data melalui berbagai visualisasi:

#### 1.4a Distribusi Fitur Numerik
Histogram dibuat untuk setiap fitur numerik dengan tambahan KDE (Kernel Density Estimation) untuk melihat bentuk distribusi data.

#### 1.4b Box Plot Berdasarkan Target
Box plot menunjukkan distribusi setiap fitur numerik berdasarkan kelas target (Membeli vs Tidak Membeli). Ini membantu melihat perbedaan karakteristik antara kedua kelas.

#### 1.4c Distribusi Fitur Kategorikal
Bar chart menampilkan jumlah data untuk setiap kategori pada fitur kategorikal seperti Month, VisitorType, dll.

#### 1.4d Count Plot Kategorikal Berdasarkan Target
Count plot dengan hue menunjukkan distribusi setiap kategori kategorikal berdasarkan target, memberikan insight tentang fitur mana yang paling berpengaruh.

#### 1.4e Correlation Heatmap
Heatmap korelasi menampilkan hubungan antar semua fitur numerik. Nilai korelasi berkisar antara -1 hingga +1:
- Nilai mendekati +1: korelasi positif kuat
- Nilai mendekati -1: korelasi negatif kuat
- Nilai mendekati 0: tidak ada korelasi

#### 1.4f Korelasi dengan Target
Bar chart horizontal menampilkan korelasi setiap fitur dengan target Revenue, diurutkan dari korelasi terendah ke tertinggi.

#### 1.4g Pair Plot Fitur Penting
Pair plot menampilkan scatter plot untuk kombinasi fitur-fitur dengan korelasi tertinggi terhadap target, dengan warna berbeda untuk setiap kelas.

#### 1.4h Violin Plot
Violin plot menggabungkan box plot dengan KDE untuk menunjukkan distribusi data yang lebih detail berdasarkan target.

#### 1.4i KDE Plot
Kernel Density Estimation plot membandingkan distribusi setiap fitur numerik antara kelas Membeli dan Tidak Membeli.

#### 1.4j Statistical Summary Visualization
Bar chart membandingkan rata-rata setiap fitur numerik antara kedua kelas target.

### 1.5 Cek Missing Value

Missing value dicek menggunakan `df.isnull().sum()` untuk setiap kolom. Missing value adalah data kosong yang dapat mempengaruhi proses pelatihan model.

### 1.6 Cek dan Hapus Data Duplikat

Data duplikat dicek menggunakan `df.duplicated().sum()`. Data duplikat dihapus menggunakan `df.drop_duplicates()` agar model tidak mempelajari data yang berulang secara berlebihan.

### 1.7 Cek Distribusi Target

Distribusi target dicek untuk mengetahui apakah dataset seimbang atau tidak:
- `False` = Pengunjung tidak melakukan pembelian
- `True` = Pengunjung melakukan pembelian

### 1.8 Balancing Data dengan Undersampling

Karena dataset memiliki distribusi kelas yang tidak seimbang (lebih banyak yang tidak membeli), dilakukan proses undersampling:
- Memisahkan data berdasarkan kelas target
- Mengambil sampel acak dari kelas mayoritas (Tidak Membeli) sebanyak jumlah kelas minoritas (Membeli)
- Menggabungkan kembali data dan mengacak urutannya

### 1.9 Visualisasi Target Setelah Balancing

Setelah undersampling, distribusi target divisualisasikan kembali untuk memastikan kedua kelas sudah seimbang.

### 1.10 Memisahkan Fitur dan Target

Dataset dipisahkan menjadi:
- **X**: Berisi seluruh fitur untuk melakukan prediksi
- **y**: Berisi target yang akan diprediksi (kolom Revenue), dikonversi ke integer (False=0, True=1)

### 1.11 Menentukan Fitur Numerik dan Kategorikal

- **Fitur Numerik**: Kolom dengan tipe data int64 atau float64
- **Fitur Kategorikal**: Kolom dengan tipe data object atau bool

### 1.12 Preprocessing Data

Preprocessing dilakukan agar data siap untuk model machine learning:
- **StandardScaler**: Menstandarkan skala fitur numerik sehingga memiliki mean=0 dan std=1
- **OneHotEncoder**: Mengubah fitur kategorikal menjadi representasi biner
- **ColumnTransformer**: Menggabungkan preprocessing numerik dan kategorikal dalam satu proses

### 1.13 Train-Test Split

Dataset dibagi dengan rasio 80:20:
- **80% data training**: Untuk melatih model
- **20% data testing**: Untuk menguji performa model

Parameter `stratify=y` memastikan distribusi target tetap seimbang di kedua subset.

### 1.14 Menyiapkan Algoritma Machine Learning

Tujuh algoritma disiapkan untuk dibandingkan:
1. **Logistic Regression**: Model linear sederhana untuk klasifikasi
2. **Decision Tree**: Pohon keputusan yang membuat aturan berdasarkan fitur
3. **Random Forest**: Kumpulan banyak pohon keputusan dengan voting
4. **Gradient Boosting**: Membangun pohon secara berurutan untuk memperbaiki error
5. **XGBoost**: Versi optimized dari gradient boosting
6. **KNN**: Mengklasifikasikan berdasarkan kedekatan dengan tetangga terdekat
7. **SVM**: Mencari hyperplane terbaik untuk memisahkan kelas

### 1.15 Training dan Evaluasi Semua Model

Setiap model dilatih dan dievaluasi menggunakan pipeline yang berisi preprocessing dan model. Evaluasi menggunakan metrik:
- **Accuracy**: Proporsi prediksi benar dari total prediksi
- **Precision**: Ketepatan prediksi positif (dari yang diprediksi membeli, berapa yang benar membeli)
- **Recall**: Kemampuan menemukan semua kasus positif (dari yang benar membeli, berapa yang terdeteksi)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Kemampuan model membedakan kelas

Model diurutkan berdasarkan F1-Score tertinggi.

### 1.15b Training XGBoost dengan SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) diterapkan pada model XGBoost:
- Mengidentifikasi data minoritas
- Membuat data sintetis berdasarkan tetangga terdekat
- Menambahkan data sintetis ke data training

### 1.16 Visualisasi Perbandingan Model

Bar chart horizontal membandingkan F1-Score semua model untuk visualisasi performa.

### 1.17 Menentukan Model Terbaik

Model dengan F1-Score tertinggi dipilih sebagai model terbaik untuk evaluasi lebih detail dan penyimpanan.

### 1.18 Evaluasi Detail Model Terbaik

Classification report menampilkan precision, recall, dan F1-Score untuk masing-masing kelas secara detail.

### 1.19 Confusion Matrix

Confusion matrix menampilkan 4 kuadran:
- **True Negative (TN)**: Tidak membeli, diprediksi tidak membeli
- **False Positive (FP)**: Tidak membeli, tapi diprediksi membeli
- **False Negative (FN)**: Membeli, tapi diprediksi tidak membeli
- **True Positive (TP)**: Membeli, diprediksi membeli

### 1.19b Pie Chart Prediksi Benar dan Salah

Pie chart menampilkan persentase prediksi benar dan salah dari model.

### 1.20 Contoh Prediksi Data Baru

Simulasi prediksi menggunakan satu data dari testing set untuk memastikan model dapat memprediksi data baru dengan probabilitas.

### 1.21 Menyimpan Model

Model terbaik disimpan dalam format `.pkl` menggunakan Joblib agar dapat digunakan kembali tanpa training ulang.

---

## Bagian 2: File `app.py`

### 2.1 Konfigurasi Halaman

Streamlit dikonfigurasi dengan:
- Judul halaman: "E-Commerce Purchase Prediction"
- Layout lebar (wide)
- Icon emoji 🛍️

### 2.2 Load Model

Model machine learning dimuat dari file `.pkl` menggunakan `@st.cache_resource` untuk caching agar tidak dimuat ulang setiap kali.

### 2.3 Header

Judul utama "Will This Visitor Buy?" dengan deskripsi singkat tentang fungsi aplikasi.

### 2.4 Mode Selector

Terdapat pilihan mode input:
- **Simple Mode**: 8 pertanyaan sederhana untuk input cepat
- **Advanced Mode**: Semua 17 fitur dapat diatur manual

### 2.5 Simple Mode

#### Session Info:
- **Month**: Pilihan bulan kunjungan (Jan-Des)
- **Weekend**: Toggle untuk kunjungan weekend

#### Visitor Type:
- **Visitor type**: Returning_Visitor, New_Visitor, atau Other

#### Visitor Behavior:
- **Product pages visited**: Slider 0-100 halaman
- **Time on product pages**: Slider 0-120 menit
- **Overall engagement**: Slider pilihan (Very low, Low, Medium, High, Very high)
- **Left after 1-2 pages**: Toggle untuk bounce behavior
- **Visit near holiday/special day**: Toggle untuk special day

#### Page Value (Fitur Paling Penting):
- Slider 0-200 dollar
- Penjelasan bahwa PageValue adalah prediktor paling penting
- Cara mendapatkannya dari Google Analytics

#### Feature Mapping:
Input sederhana dipetakan ke fitur teknis model:
- `engagement_map`: Konversi level engagement ke Administrative dan Informational pages
- `bounce` dan `exit_r`: Dihitung berdasarkan toggle "left quickly"
- `special_val`: Dihitung berdasarkan toggle "special day"
- Technical defaults (OS, Browser, Region, Traffic) menggunakan nilai paling umum

### 2.6 Advanced Mode

Semua 17 fitur dapat diatur secara manual:

#### Page Visits:
- Admin pages, Info pages, Product pages (jumlah)
- Durasi masing-masing dalam detik

#### Behavior:
- Bounce rate (0-1)
- Exit rate (0-1)
- Page value (nilai numerik)
- Special day proximity (0-1)

#### Session Context:
- Month dan Weekend

#### Visitor Type:
- Returning_Visitor, New_Visitor, Other

#### Technical (anonymized IDs):
- OS (1-8)
- Browser (1-13)
- Region (1-9)
- Traffic type (1-20)

### 2.7 Predict Button

Tombol "Predict Purchase" memicu proses prediksi:
- `model.predict()`: Menghasilkan kelas (0 atau 1)
- `model.predict_proba()`: Menghasilkan probabilitas pembelian

### 2.8 Display Result

Hasil ditampilkan dengan:
- **Success/Warning**: Berdasarkan prediksi (Membeli/Tidak Membeli)
- **Metric**: Probabilitas pembelian dalam persentase
- **Progress bar**: Visualisasi confidence
- **Expander**: Menampilkan semua fitur yang dikirim ke model

### 2.9 Footer

Informasi model yang digunakan: XGBoost dengan SMOTE class balancing, dilatih pada UCI Online Shoppers Intention dataset.

---

## Ringkasan Alur Kerja

1. **Data Loading**: Dataset dimuat dari CSV
2. **EDA**: Data dianalisis dan divisualisasikan
3. **Data Cleaning**: Missing values dan duplikat dihapus
4. **Balancing**: Dataset diseimbangkan menggunakan undersampling
5. **Preprocessing**: Data dinormalisasi dan di-encode
6. **Training**: 8 model dilatih dan dibandingkan
7. **Evaluation**: Model terbaik dipilih berdasarkan F1-Score
8. **Deployment**: Model disimpan dan digunakan dalam aplikasi Streamlit

---

## Fitur-Fitur dalam Dataset

| Fitur | Deskripsi |
|-------|-----------|
| Administrative | Jumlah halaman admin yang dikunjungi |
| Administrative_Duration | Waktu di halaman admin (detik) |
| Informational | Jumlah halaman informasi yang dikunjungi |
| Informational_Duration | Waktu di halaman informasi (detik) |
| ProductRelated | Jumlah halaman produk yang dikunjungi |
| ProductRelated_Duration | Waktu di halaman produk (detik) |
| BounceRates | Persentase bounce (keluar setelah 1 halaman) |
| ExitRates | Persentase exit (halaman terakhir sebelum keluar) |
| PageValues | Nilai rata-rata halaman berdasarkan konversi |
| SpecialDay | Kedekatan dengan hari khusus (0-1) |
| Month | Bulan kunjungan |
| OperatingSystems | ID sistem operasi |
| Browser | ID browser |
| Region | ID region |
| TrafficType | ID tipe traffic |
| VisitorType | Tipe pengunjung (Returning/New/Other) |
| Weekend | Kunjungan di weekend atau tidak |
| Revenue | Target: apakah pengunjung membeli atau tidak |

---

## Kesimpulan

Proyek ini berhasil membangun sistem prediksi pembelian e-commerce menggunakan machine learning. Model XGBoost dengan SMOTE memberikan performa terbaik. Aplikasi Streamlit menyediakan antarmuka yang user-friendly untuk melakukan prediksi dengan dua mode input: sederhana dan lanjutan.