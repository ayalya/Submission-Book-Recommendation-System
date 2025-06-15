# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Dengan meningkatnya penggunaan internet, seharusnya informasi mengenai buku telah tersebar luas. Namun kenyataannya, di Indonesia sendiri minat baca masih rendah dengan rasio pembaca buku 1 berbanding 1000 orang yang memiliki kebiasaan membaca. Penggunaan internet juga merupakan generasi muda dan tergolong produktif. Keadaan ini dapat menjadi peluang dan tantangan untuk meningkatkan minat baca masyarakat Indonesia[[1]](#ref1).

Jumlah buku yang terus bertambah yang dicatat oleh Perpustakaan Nasional Rebuplik Indonesia penomoran ISBN pada tahun 2017 sekitar 67 ribu judul buku dan tahun 2022 mencapai 642 ribu judul buku[[2]](#ref2). Banyaknya informasi mengenai judul buku justru menyulitkan pengguna untuk menemukan buku yang sesuai dengan minat dan preferensi pribadi. Maka dari itu, diperlukan suatu penyaring informasi mengenai buku yang sesuai baik berdasarkan judul buku, penulis, hingga genre buku. Peningkatan penggunaan internet dan bertambahnya jumlah buku, menjadikan peluang memadukan keduanya dengan memanfaatkan algoritma machine learning dalam membuat sebuah sistem rekomendasi buku. Berdasarkan penelitian yang telah dilakukan [[3]](#ref3), dengan mengimplementasikan sistem rekomendasi pada buku di *website* perpustakaan, membantu memberikan rekomendasi sesuai dengan minat pengguna.

## Business Understanding

### Problem Statements

Berdasarkan permasalahan di atas, bagaimana didapatkan permasalahan, yaitu:
- Bagaimana cara membangun sistem rekomendasi buku berdasarkan preferensi pribadi pengguna terhadap identitas buku (content-based filtering)?
- Bagaimana cara membangun sistem rekomendasi buku berdasarkan kesamaan perilaku pengguna yang serupa (collaborative-filtering)?

### Goals

Untuk menjawab permasalahan di atas, maka tujuan dari proyek ini, yaitu:
- Membangun sistem rekomendasi buku dengan pendekatan content-based filtering untuk memberikan saran buku berdasarkan informasi judul buku yang dimasukkan oleh user.
- Membangun sistem rekomendasi buku dengan pendekatan collaborative filtering untuk memberikan saran buku berdasarkan perilaku dan penilaian orang lain.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution Approach
Tahapan untuk mencapai tujuan pada proyek ini adalah sebagai berikut:
1. Solution Statement 1: Content-Based Filtering
    Pendekatan ini fokus kepada fitur-fitur buku, seperti ISBN, judul buku, penulis buku, penerbit, dan tahun diterbitkannya buku. Pendekatan ini menggunakan Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer untuk megubah dokumen menjadi representasi matriks dan mengukur seberapa mirip antara dua vektor menggunakan cosine similarity. Hasilnya adalah rekomendasi buku yang mirip dengan buku yang pernah disukai pengguna.
2. Solution Statement 2: Collaborative filtering
    Pendekatan ini memanfaatkan pola yang ditandai dengan penilaian (rating) menggunakan model neural network collaborative filtering, yaitu memanfaatkan embedding dengan neural network. Model ini memberikan representasi pengguna dan buku untuk memberikan rekomendasi buku berdasarkan kesamaan pola dengan pengguna lain.

## Data Understanding

Dataset yang digunakan pada pryek ini adalah dataset untuk merekomendasikan buku dengan judul Book Recommendation Dataset yang diambil melalui platform [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Data ini berisikan informasi tentang buku, rating, dan demografi pengguna. Terdiri dari tiga file csv, yaitu `Books.csv`, `Rating.csv`, dan `Useres.csv`.

Dapat diketahui bahwa:

1. Variabel `Books` memiliki jumlah data 271.360 data dengan 8 kolom, yaitu:
  - `ISBN`, nomor identifikasi buku unik yang diberikan oleh Badan ISBN Internasional.
  - `Book-Title`, judul buku.
  - `Book-Author`, Penulis buku.
  - `Year-Of-Publication`, Tahun buku diterbitkan.
  - `Publisher`, Penerbit buku.
  - `Image-URL-S`, diperoleh dari web Amazon Web Service (AWS) untuk menautkan ke gambar sampul kecil.
  - `Image-URL-M`, diperoleh dari web AWS untuk menautkan ke gambar sampul sedang.
  - `Image-URL-L`, diperoleh dari web AWS untuk menautkan ke gambar sampul besar.
2. Dataset `Ratings`, memiliki jumlah data 340.556 data dengan 3 kolom, yaitu:
  - `User-ID`, ID pengguna yang dianonimus dan petakan dengan nilai integer.
  - `ISBN`, nomor identifikasi buku yang diberikan rating oleh users.
  - `Book-Rating`, rating buku bernilai 0 sampai 10 (semakin tinggi nilai, semakin besar apresiasi).
3. Dataset `Users`, memiliki jumlah data 278.858 data dengan 3 kolom, yaitu:
  - `User-ID`, ID pengguna yang dianonimuskan dan dipetakan dengan nilai integer yang memberikan informasi pada data user.
  - `Location`, data demografi lokasi user jika diizinkan oleh pengguna.
  - `Age`, data demografi berupa umur pengguna jika diizinkan oleh pengguna.

### Tipe Data Setelah Dilakukan Pembersihan

1. **Book.csv**
    |Kolom                  | Tipe Data |
    |ISBN                   | object    |
    |Book-Title             | object    |
    |Book-Author            | object    |
    |Year-Of-Publication    | object    |
    |Publisher              | object    |
    |Image-URL-S            | object    |
    |Image-URL-M            | object    |
    |Image-URL-L            | object    |

2. **Users.csv**
    |Kolom                  | Tipe Data |
    |User-ID                | int64     |
    |ISBN                   | object    |
    |Book-Rating            | int64     |

3. **Users.csv**
    |Kolom                  | Tipe Data |
    |User-ID                | int64     |
    |Location               | object    |
    |Age                    | int64     |

**Rubrik/Kriteria Tambahan (Opsional)**:

Berikut tahapan yang dilakukan pada explorasi dan visualisasi data:
1. **Penulis Buku dengan Buku Terbanyak**

    <div>
        <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/Top10PenulisBuku.png">
    </div>

    *Gambar 1, Visualisasi 10 Penulis Teratas dengan Buku Terbanyak*

    Penulis dengan judul buku terbanyak ditulis oleh Agatha Christie yang mencapai lebih dari 600 buku, diikuti oleh William Shakespeare dan Stephen King. Untuk mencapai penulis dengan buku terbanya, setidaknya harus menulis 300 buku. Penulis-penulis berikut cukup dikenal publik dan produktif, sehingga karyanya ditunggu-tunggu oleh pembaca.

2. **Tahun dengan Jumlah Buku Terbit Terbanyak**

    <div>
        <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/Top10TahunTerbit.png">
    </div>

    *Gambar 2, Grafik Batang Tahun dengan Jumlah Buku Terbit Terbanyak*

    Akhir tahun 90an hingga awal tahun 2000an merupakan puncaknya membaca buku. Hal ini ditandai dengan buku-buku yang terbit di tahun 1998 hingga 2002 merupakan tahun dengan jumlah buku yang terbit tinggi, hingga lebih dari 12 ribu buku terbit pada tahun tersebut.

3. **Distribusi Buku yang diberi Rating**

    <div>
        <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/JumlahRating.png">
    </div>

    *Gambar 3, Grafik Batang Rating Terbanyak*

    Rating yang diberikan pada buku berkisar 0 hingga 10, dimana semakin tinggi rating diberikan, maka semakin besar buku direkomendasikan kepada pengguna lain. Pada visualisasi persebaran rating di atas, buku yang diberi rating 0 atau bisa diindikasikan tidak diberi rating merupakan angka tertinggi. Sehingga kebanyakan buku yang diberi rating, mencakup di angka 8, 7, dan 10.

4. **Distribusi Demografi Usia**

    <div>
        <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/DemografiUsia.png">
    </div>

    *Gambar 4, Garfik Batang Persebaran Usia Users*

    Pada distribusi usia pengguna, usia kebanyakan pengguna berada di awal 20an hingga pertengahan usia 30an. Pengguna paling banyak berada pada usia pertangan 20an, bahkan usia 25 merupakan pengguna tertinggi. Distribusi pada persebaran usia ini miring ke kanan atau right-skewed.

## Data Preparation
Pada tahapan ini, dilakukan proses persiapan data untuk memastikan data bersih, konsisten, sesuai dengan kebutuhan algoritma. Persiapan data dibagi menjadi dua jenis berdasarkan kategori algoritma.

### Data Preparation Pendekatan Content-Based Filtering
Pada pendekatan content-based filtering, variabel yang akan digunakan, yaitu `ISBN`, `Book-Title`, `Book-Author`, dan `Publisher`. Berikut tahapan yang dilakukan pada tahap data preparation dengan pendekatan content-based filtering:

#### 1. Menghapus nilai yang kosong
Nilai kosong pada data yang akan digunakan terlihat pada `Book-Author`, `Publisher`, dan `Year-Of-Publication`. Jumlah data yang kosong terhitung sedikit dan baris yang kosong akan dihapus karena tidak mempengaruhi informasi pada keseluruhan data.

#### 2. Menghapus Nilai Invalid
Pada tahapan eksplorasi data, didapatkan bahwa terdapat nilai yang tidak valid karena salah input. Nilai yang tidak valid ini terdapat kesalahan pada fitur `Year-Of-Publication` menjadi nilai `Publisher` yang berisi "DK Publishing Inc" dan "Gallimard". Jumlah data yang mengandung nilai invalid adalah tiga data dan baris yang mengandung kedua nilai tersebut dihapus. Maka dari itu, `Year-Of-Publication` dapat diganti tipe datanya menjadi integer.

#### 3. Standarisasi ISBN
ISBN merupakan nilai unik pada setiap judul buku. Jika terdapat nilai ISBN yang sama pada dua buku atau lebih, dapat dikatakan bias dan nilainya tidak valid. Panjang karakter ISBN adalah 10 hingga 13 dan diawali dengan huruf. Awalnya nilai `ISBN` adalah 270.144 dan `Book-Title` sebanyak 241.065. Setelah dilakukan standarisasi data, didapatkan nilai unik `ISBN` dengan `Book-Title` memiliki panjang yang sama, yaitu 247.936 baris.

#### 4. Penggabungan Data
Data yang telah dibersihkan digabung menjadi DataFrame dengan kolom `ISBN`, `Book-Title`,`Book-Author`, `Publisher`, dan `Year-Of-Publication` dengan jumlah baris 247.936. Pada dataset ini juga ISBN akan dijadikan key-value pada tahapan penyatuan dan pencocokan data pada tahap selanjutnya. Karena data dengan jumlah tersebut sangat banyak, untuk memanfaatkan komputasi dibatasi data yang akan digunakan sebanyak 20.000 baris data.

#### 5. TF-IDF Vectorization
Fitur-fitur di atas diubah menjadi representasi numerik dan matriks menggunakan library `sklearn.feature_extraction.text`. TF merupakan proses mengukur seberapa sering kata-kata muncul dan IDF mengukur seberapa unik atau langka sebuah kata dari dokumen. Pada proses ini juga dilakukan proses penghapusan stopwords dalam Bahasa Inggris pada fitur `Book-Author`

#### 6. Cosine Similarity
Untuk mengukur kesamaan antara dua vektor, digunakan perhitungan pada matriks TF-IDF sebelumnya menggunakan cosine similarity yang diambil dari library `sklearn.metrics.parwise`.


### Data Preparation Pendekatan Collaborative Filtering
Pada pendekatan ini, data yang akan digunakan menggunakan fitur-fitur pada data `Users.csv` dengan memanfaatkan data rating dan informasi pengguna. Sebelumnya, proses persiapan data rating perlu ditransformasikan ke dalam bentuk matriks numerik untuk memudahkan model belajar. Dilakukan proses encoder pada fitur `User-ID` dan `ISBN` yang menjadi `book_title` ke dalam bilangan bulat dan pemetaan ke dalam dataframe yang telah rapih sebelumnya. Berikut adalah tahapan-tahapan persiapan data untuk pendekatan collaborative filtering:

#### 1. Filter Data Rating Positif
Karena data rating banyak bernilai 0 atau tidak memberikan rating nyata, jadi hanya rating yang lebih dari 0 akan digunakan.

#### 2. Melakukan Encoding
Proses encoder atau pengkodean fitur dilakukan pada `User-ID` ke `ISBN` untuk menjadi representasi bilangan bulat. Selanjutnya dilakuakan pencocokan dengan dataset yang sebelumnya telah rapih dengan `ISBN`. Sehingga fitur yang akan digunakan untuk pendekatan collaborative filtering adalah `Book-Rating`, `user` (nilai `User-ID` yang telah dilabeli), dan `Book-Title` yang telah dilabeli.

#### 3. Pemisahan Data Latih dan Data Uji
Membuat variabel baru khusus untuk Collaborative Filtering. Data sampel pada model ini adalah `user` dan `book_title`, sedangkan targetnya adalah `Book-Rating`. Pemisahan data dilakukan membagi data latih menjadi 80% dan data uji atau validasi menjadi 20%.


## Modeling
Pada bagian ini model sistem rekomendasi dibuat untuk menyelesaikan permasalahan dalam meningkatkan penjualan buku dan meningkatkan performa penjualan.

### 1. Content-Based Filtering

#### Cara Kerja
Tahapan pertama pengembangan model pada tahapan ini menggunakan teknik Content-Based Filtering. Teknik ini merupakan sebuah pendekatan dalam sistem rekomendasi dengan memanfaatkan informasi atau konten dari item atau pengguna untuk membuat rekomendasi. Ide dasar pendekatan ini adalah dengan mencocokan preferensi pengguna dengan karakteristik atau konten dari item yang pernah dilihat atau disukai pengguna sebelumnya. Misalnya, jika pengguna pernah menyukai karya atau membeli buku novel "The Woman in The Moon and Other Tales of Forgotten Heroines" dan buku tersebut memiliki fitur penulis buku bernama "James Riordan", maka sistem akan mencarikan dan merekomendasikan buku lain dengan fitur yang serupa berbentuk rekomendasi top-N kepada pengguna. Pada pendekatan ini, sistem akan menampilkan Top 5 buku rekomendasi dengan langkah-langkah:
1. Menggabungkan Fitur `ISBN`,`Book-Title`, `Book-Author`, dan `Publisher`.
2. Mengubah representasi teks menjadi vektor menggunakan TF-IDF.
3. Mengukur kesamaan antar dua vektor TF-IDF antar buku menggunakan cosine similarity.
4. Menyajikan rekomendasi buku yang serupa dengan buku yang disukai pengguna.

#### Parameter
- Vectorizer, TF-IDF menggunakan `stop_words='english'`
- Similarity Metric, default dan flatten()

#### Kelebihan
- Kemampuan memberikan rekomendasi lebih personal.
- Tidak bergantung terhadap pengguna lain

#### Kekurangan
- Rekomendasi hanya terbatas keada konten dan cakupan lebih kecil.
- Tidak menyarankan item yang lebih variatif di luar sejarah preferensi pengguna.

#### Hasil Prediksi
Pada tahap ini, judul buku yang dicari adalah "The Woman in the Moon and Other Tales of Forgotten Heroines" karya James Riordan. Model content-based filtering mempu memberikan rekomendasi 5 buku dengan konten yang mirip

<div>
    <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/PredictContentBased.png">
</div>

*Gambar 5, Top 5 Hasil Prediksi Content-Based Filtering*

### 2. Collaborative Filtering
Collaborative Filtering merupakan salah satu metode sistem rekomendasi dengan memprediksi preferensi pengguna terhadap suatu item berdasarkan informasi dari pengguna lain (kolaborasi). Ide dasar teknik ini adalah pengguna dengan preferensi yang sama di masa lalu cenderung memiliki preferensi yang sama di masa depan. Hasil pada model ini akan memberikan rekomendasi sejumlah judul yang sesuai dengan preferensi pengguna berdasarkan rating yang diberikan.

Tahapan yang dilakukan:
1. Membagi dataset menjadi data training 80% dan testing 20%.
2. Training model dengan menghitung kecocokan antar pengguna menggunakan fitur `user` dan `book_title` dengan menggunaka teknik embedding. Layer neural network diberikan pada `user`, `book_title`, dan dot product keduanya.
3. Nilai kecocokan diatur pada skala [0,1] dengan fungsi aktivasi sigmoid.

#### Parameter
- Hidden layers = 4 layer embedding
- Output later = 1 unit sigmoid
- Loss function = Binary Crossentropy
- Optimizer = Adam
- Epochs = 5
- batch size = 64

#### Kelebihan
- Memberikan rekomendasi lebih fleksibel dan menangkap hubungan non-linear antara user dan item.
- Cocok untuk data yang besar dan kompleks.

#### Kekurangan
- Tidak dapat merekomendasikan item yang sangat berbeda dari yang telah disukai.
- Memerlukan banyak data dan waktu pelatihan.

#### Hasil Prediksi
Pada tahapan ini, model memberikan informasi pengguna dengan `User-ID` 8695 telah memberikan rating pada buku, berikut:

<div>
    <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/CFRatingbyUsers.png">
</div>

*Gambar 6, Buku-Buku yang Telah Diberikan Rating oleh Pengguna dengan User-ID 8695*

Lalu dia akan mendapatkan rekomendasi 10 buku berikut:

<div>
    <img src="https://raw.githubusercontent.com/ayalya/Submission-Book-Recommendation-System/main/asset/CFReccomendationtoUser.png">
</div>

*Gambar 7, Buku-Buku yang akan Direkomendasikan kepada Pengguna dengan User-ID 8695*

## Evaluation

Tahapan evaluasi akan dibagi berdasarkan pendekatan sistem rekomendasinya karena memiliki matriks evaluasi yang berbeda berdasarkan karakter outputnya.

#### 1. Content-Based Filtering

Pendekatan ini memberikan rekomendasi Top-N buku berdasakan kemirpan konten, maka dari itu metrik yang digunakan adalah Precision, Recall, dan F1-Score.

- **Precision**, memberikan model yang benar-benar relevan.

    ![Precision](https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Precision}=\frac{TP}{TP+FP})

- **Recall**, memberikan informasi berapa item yang berhasil direkomendasikan.

    ![Recall](https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Recall}=\frac{TP}{TP%20+%20FN})

- **F1-Score**, harmonisasi antara precision dan recall.

   ![F1 Score](https://latex.codecogs.com/png.image?\dpi{120}&space;F1\_Score=2\cdot\frac{Precision\cdot%20Recall}{Precision+Recall})

**Hasil**

*Tabel 1, Evaluasi Matriks Content-Based Filtering*

Precision  |    Recall  | F1 Score  
:------:|:------:|:-----
1.0      |     1.0    |      1.0  

Artinya, hasil prediksi positif berhasil dan model menangkap semua data positif. Sehingga nilai F1 juga sempurna. Namun model dengan hasil sempurna patut diwaspadai dikhawatirkan adanya data yang tidak seimbang atau data yang bocor.

#### 2. Collaborative Filtering

Pendekatan ini menggunakan evaluasi matriks Mean Squared Error dan Root Mean Squared Error.

- **Mean Squared Error (MSE)**
    Digunakan untuk menghitung nilai selisih antara rating prediksi dan rating aktual. Metirk ini memberikan gambaran intuisif tentang kesalahan rata-rata pada saat prediksi.

    ![MAE](https://latex.codecogs.com/png.image?\dpi{150}&space;\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}\left|\hat{y}_i-y_i\right|)


- **Root Mean Squared Error (RMSE)**
    Digunakan untuk mengukur rata-rata kesalahan kuadrat dari prediksi terhadap nilai aktual. Semakin kecil nilai RMSE, maka semakin baik performanya.

    ![RMSE](https://latex.codecogs.com/png.image?\dpi{150}&space;\text{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2})


**Hasil**

*Tabel 2, Evaluasi Matriks Content Filtering*

|        Data       |   MAE   |  RMSE   |
|:-----------------:|:-------:|:-------:|
| **Latih**         | 0.0424  | 0.2059  |
| **Uji**           | 0.0804  | 0.2836  |

Artinya, model lebih baik melakukan pelatihan dibandingkan dengan pengujian. Meskipun demikian, nilainya data latih dan data uji tidak berbeda jauh yang menandakan model tidak terjadi overfitting dan menggeneralisasi dengan baik.

Evaluasi pada kedua model menandakan bahwa model dapat berjalan dengan baik, juga memberikan rekomendasi yang berkualitas dan kinerjanya terukur.

Berikut adalah jawaban dari problem statement:

#### 1. Apakah Sudah Menjawab Problem Statement?
- Problem Statement 1: Bagaimana cara membangun sistem rekomendasi buku berdasarkan preferensi pribadi pengguna terhadap identitas buku (content-based filtering)?

    Jawaban: Terjawab, pendekatan Content-Based Filtering dibangun dengan teknik TF-IDF dan cosine similarity dapat memberikan rekomendasi yang relevan berdasarkan informasi konten buku, seperti judul buku, penulis, penerbit, dan tahun terbut. Pendekatan ini bergantung kepada kontek buku dan tidak bergantung kepada pengguna lain, sehingga efektif untuk pengguna baru.

- Problem Statement 2: Bagaimana cara membangun sistem rekomendasi buku berdasarkan kesamaan perilaku pengguna yang serupa (collaborative-filtering)?
    Jawaban: terjawab. Pendekatan collaborative-filtering memanfaatkan data interaksi pengguna dengan buku, seperti rating yang diberikan pada buku. Pendekatan yang dilakukan menggunakan Neural Collaborative Filtering sehingga diperlukan data dengan tipe data numerik. Hasilnya sistem rekomendasi ini memberikan prediksi rating yang cukup akurat berdasarkan pola perilaku pengguna lain yang serupa.

#### 2. Apakah Berhasil Mencapai Setiap Goals yang Diharapkan?
- Goal 1: Membangun sistem rekomendasi buku dengan pendekatan content-based filtering untuk memberikan saran buku berdasarkan informasi judul buku yang dimasukkan oleh user.

    Jawaban: tercapai. Dibuktikan dengan metrik evaluasi, seperti Precision, Recall, dan F1 Score memberikan hasil yang baik dalam memberikan rekomendasi secara personal dan berdasarkan kontennya.

- Goal 2: Membangun sistem rekomendasi buku dengan pendekatan collaborative filtering untuk memberikan saran buku berdasarkan perilaku dan penilaian orang lain.

    Jawaban: Tercapai. Dibuktikan dengan nilai MSE dan RMSE yang rendah, artinya memiliki kesalahan yang kecil dalam memberikan rekomendasi kepada pengguna berdasarkan data interaksi pengguna lain yang serupa.

#### 3. Apakah Solusi Statement yang Direncakanan Berdampak?

- Solusi Conten-Based Filtering:
    Berdampak. Sistem ini memberikan rekomendasi sesuai dengan langkah-langkah yang telah didefinisikan. Hasilnya juga sesuai dengan yang diinginkan. Solusi ini membantu dalam memberikan rekomnedasi bagi pengguna yang baru.

- Solusi Collaborative Filtering:
    Sangat berdampak. Sistem ini mampu menangkap pola kompleks interaksi pengguna dan mampu meminimalkan kesalahan dalam merekomendasikan. Dengan menggunakan embedding jaringansaraf, sistem dapat menangkap pola tersembunyi yang tidak biasa ditangkap dengan pendekatan yang sederhana.


## Kesimpulan

Proyek ini mampu memberikan sistem rekomendasi dengan pendekatan content-based filtering dan collaborative filtering untuk memberikan pengguna pengalaman dalam menemukan buku yang sesuai dengan preferensi mereka. Sehingga kesimpulan yang didapatkan pada proyek ini, yaitu:

1. Sistem rekomendasi berhasil dibangun menggunakan dua pendekatan, yaitu:
    - Content-Based Filtering, menggunakan TF-IDF dalam merepresentasikan vektor dan cosine similarity untuk mengukur kemiripan antar buku berdasarkan informasi kontennya.
    - Collaborative Filtering, menggunakan pendekatan neural network, mampu memprediksi interaksi pengguna buku berdasarkan informasi pengguna dan menangkap pola yang rumit.
2. Hasil evaluasi pada kedua model, baik content-based dengan F1 Score dan collaborative filtering dengan RMSE dan MSE memberikan performa yang baik sehingga kedua model mempu menangani pola interaksi pengguna yang kompleks.
3. Keterikatan model dengan business understanding. Baik problem statement dan tujuan bisnis mampu dicapai dan solusi yang diajukan berdampak kepada pengembangan sistem.
4. Sistem rekomendasi ini dapat meningkatkan kepuasan pengguna dengan menyediakan kepuasn terhadap pengguna dengan menyediakan rekomendasi buku yang sesuai minat sehingga mendorong eksplorasi terhadap koleksi buku secara luas.

## Referensi
<a name="ref1"></a>[1] R. Ardiansyah, M. Ari Bianto, and B. D. Saputra, “Sistem Rekomendasi Buku Perpustakaan Sekolah menggunakan Metode Content-Based Filtering”, CoSciTech, vol. 4, no. 2, pp. 510-518, Oct. 2023.

<a name="ref2"></a>[2] H. Dharmawan, Tukino, S. Shofiah Hilabi, and I. Karniawulan, “SISTEM REKOMENDASI BUKU DENGAN METODE K-NEAREST NEIGHBOR (K-NN) PADA GRAMEDIA ”, zn, vol. 5, no. 1, pp. 16–25, Jan. 2023..

<a name="ref3"></a>[3] MRosyadS., MahendraD., and AzizahN., “SISTEM REKOMENDASI BUKU DI PERPUSTAKAAN DAERAH JEPARA MENGGUNAKAN METODE ITEM-BASED COLLABORATIVE FILTERING”, Biner : Jurnal Ilmiah Informatika dan Komputer, vol. 2, no. 2, pp. 76-81, Jul. 2023.
