# Prediksi Pembelian E-Commerce

Memprediksi apakah pengunjung website akan melakukan pembelian menggunakan machine learning.

## Struktur Proyek

```
├── App/                   # Aplikasi web Streamlit
│   └── app.py
├── Data/                  # Direktori dataset
│   └── online_shoppers_intention.csv
├── Models/                # File model terlatih
│   └── model_prediksi_pembelian_ecommerce.pkl
├── Outputs/               # Hasil evaluasi
├── Notebooks/             # Jupyter notebooks
│   └── prediksi_pembelian_ecommerce.ipynb
├── requirements.txt       # Dependensi Python
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Penggunaan

### Jalankan Aplikasi Web
```bash
streamlit run App/app.py
```


## Dataset
 
- **Nama**: Online Shoppers Intention Dataset
- **Sumber**: UCI Machine Learning Repository
- **Deskripsi**: Dataset ini berisi data sesi pengunjung website e-commerce, mencakup metrik perilaku, informasi sesi, dan tipe pengunjung.
- **Target**: Kolom `Revenue` (True jika sesi berakhir dengan pembelian, False jika tidak).
- **Jumlah Fitur**: 17 fitur perilaku pengunjung.
 
## Detail Model

- **Algoritma**: Gradient Boosting dengan SMOTE untuk balancing kelas
- **Fitur**: 17 metrik perilaku pengunjung
- **Target**: Prediksi pembelian (0 = Tidak Beli, 1 = Beli)
- **Data pelatihan**: Dataset UCI Online Shoppers Intention

---

## Panduan Lapangan — Apa itu setiap kolom? Dari mana mendapatkannya?

Alat ini memprediksi kemungkinan pembelian menggunakan **17 data** tentang sesi pengunjung.
Berikut penjelasan dalam bahasa Indonesia untuk setiap kolom, cara mendapatkannya, dan bagaimana datanya dikumpulkan.

### 📅 Info Sesi

**Bulan**  
*Apa itu:* Bulan kalender ketika kunjungan terjadi.  
*Cara mendapatkannya:* Google Analytics → Audience → Overview → Primary Dimension: Month.  
*Mengapa penting:* Perilaku belanja berubah sesuai musim. November (Black Friday) dan Desember
(Natal) memiliki tingkat pembelian jauh lebih tinggi dibandingkan Februari.

**Akhir pekan**  
*Apa itu:* Apakah kunjungan terjadi pada hari Sabtu atau Minggu.  
*Cara mendapatkannya:* Google Analytics menghitung ini secara otomatis dari timestamp sesi.  
*Mengapa penting:* Pembelanja akhir pekan mungkin memiliki niat berbeda dengan penjelajah hari kerja.

---

### 👤 Tipe Pengunjung

**Tipe pengunjung**  
*Apa itu:* Apakah orang ini pernah mengunjungi website sebelumnya.  
- **Returning Visitor** — pernah mengunjungi sebelumnya, mungkin pernah membeli sebelumnya.  
- **New Visitor** — pertama kali di website Anda.  
- **Other** — kategori langka (misalnya bot, tidak terklasifikasi).  

*Cara mendapatkannya:* Google Analytics → Audience → Behavior → New vs Returning.  
*Mengapa penting:* Pengunjung yang kembali lebih mungkin membeli karena sudah percaya pada merek.
Sekitar 77% pembeli dalam data pelatihan adalah pengunjung yang kembali.

---

### 📄 Kunjungan Halaman

**Halaman produk yang dikunjungi**  
*Apa itu:* Berapa banyak halaman daftar produk atau detail produk yang dilihat pengunjung.  
*Cara mendapatkannya:* Google Analytics → Behavior → Site Content → All Pages. Filter halaman yang
mengandung URL produk Anda (misalnya `/product/`, `/p/`). Hitung sesi yang mengunjungi halaman-halaman ini.  
*Mengapa penting:* Lebih banyak tampilan halaman produk = niat beli lebih tinggi. Pembeli dalam data pelatihan
mengunjungi rata-rata **48 halaman produk**, sedangkan bukan pembeli hanya **29**.

**Waktu di halaman produk (menit)**  
*Apa itu:* Total waktu pengunjung menghabiskan membaca halaman produk.  
*Cara mendapatkannya:* Google Analytics → Behavior → Site Content → All Pages → Average Time on Page.
Kalikan dengan jumlah halaman produk yang dikunjungi.  
*Mengapa penting:* Waktu lebih lama di halaman produk menunjukkan pengunjung sedang meneliti,
membandingkan, atau membaca ulasan — sinyal beli yang kuat.

**Tingkat Keterlibatan (Overall Engagement)**
*Apa itu:* Ringkasan seberapa aktif pengunjung menjelajahi halaman admin dan informasi.
*Cara mendapatkannya:* Di aplikasi, pilih dari "Very Low" hingga "Very High".
*Mengapa penting:* Menggantikan input manual untuk jumlah halaman admin dan info. Keterlibatan tinggi biasanya berarti pengunjung sedang mencari detail akun atau bantuan sebelum membeli.


---

### 📈 Metrik Perilaku

**Keluar dengan Cepat (Bounced)**
*Apa itu:* Apakah pengunjung meninggalkan situs setelah hanya melihat 1-2 halaman.
*Cara mendapatkannya:* Toggle "Left after 1–2 pages" di aplikasi.
*Mengapa penting:* Mengatur `Bounce Rate` dan `Exit Rate` secara otomatis. Pembeli hampir tidak pernah bounce (rata-rata 0.5%), sedangkan non-pembeli lebih sering melakukannya.


**Page Value** ⭐ *(paling penting)*  
*Apa itu:* Nilai moneter rata-rata suatu halaman, dihitung dari semua sesi yang
termasuk halaman tersebut **dan berakhir dengan pembelian**. Halaman dengan Page Value adalah yang
biasanya mengarah ke penjualan.  
*Cara mendapatkannya:* Google Analytics → Behavior → Site Content → All Pages →
lihat kolom **"Page Value"**.  
- Jika suatu halaman memiliki Page Value = $20, berarti pengunjung yang melihat halaman tersebut
  rata-rata menghabiskan $20 di situs Anda.  
- Halaman dengan Page Value $0 tidak pernah mengarah ke pembelian dalam data.  
*Mengapa penting:* Ini adalah **prediktor terkuat** dalam model. Bahkan Page Value kecil
($5+) secara dramatis meningkatkan probabilitas pembelian.  
*Cara pengumpulan:* Google Analytics melacak setiap page view, dan ketika pengunjung menyelesaikan
transaksi, ia memberikan sebagian nilai transaksi tersebut kembali ke semua halaman yang
dikunjungi pengunjung selama sesi tersebut.

**Kunjungan Hari Spesial**
*Apa itu:* Apakah kunjungan terjadi mendekati hari besar (Black Friday, Natal, dll.).
*Cara mendapatkannya:* Toggle "Visit near a holiday" di aplikasi.
*Mengapa penting:* Hari spesial secara signifikan meningkatkan niat beli.

---

### 🔧 Detail Teknis (Dianonimisasi)

**Sistem operasi, Browser, Region, Jenis traffic**  
*Apa itu:* ID yang dianonimisasi mewakili tipe perangkat pengunjung, browser,
region geografis, dan bagaimana mereka menemukan website (pencarian, langsung, sosial, dll.).  
*Cara mendapatkannya:* Google Analytics → Audience → Technology (OS, Browser) atau
Geo → Region, atau Acquisition → All Traffic → Channels (Traffic type).  
*Mengapa penting:* Ini adalah faktor minor. Model menggunakannya sebagai penentu akhir.
**Anda bisa aman menggunakan default** — sudah disetel ke nilai paling umum
dan mengubahnya tidak akan secara signifikan mempengaruhi prediksi.

---

### 💡 Ringkasan Cepat

| Input Aplikasi | Fitur Model yang Terpengaruh | Dampak pada prediksi |
|---|---|---|
| **Page Value** | PageValues | ⭐⭐⭐⭐⭐ Sangat tinggi |
| **Product pages visited** | ProductRelated | ⭐⭐⭐⭐ Tinggi |
| **Time on product pages** | ProductRelated_Duration | ⭐⭐⭐ Sedang |
| **Left quickly (Bounced)** | BounceRates, ExitRates | ⭐⭐⭐ Sedang |
| **Visitor type** | VisitorType | ⭐⭐ Sedang |
| **Overall engagement** | Administrative, Informational | ⭐⭐ Sedang |
| **Month** | Month | ⭐ Rendah |
| **Weekend** | Weekend | ⭐ Rendah |
| **Special day** | SpecialDay | ⭐ Rendah |
| **Technical defaults** | OS, Browser, Region, Traffic | ⭐ Sangat rendah |