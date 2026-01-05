# Tugas Decision Tree - Prediksi Survival Titanic

👥 Kontributor
Nama: Abdul Hafizh Ar Rasyid

NIM: 231011403352

Mata Kuliah: Machine Learning

Dosen: AGUNG PERDANANTO S.Kom, M.Kom

## Deskripsi
Implementasi algoritma **Decision Tree** untuk prediksi survival penumpang Titanic. Proyek ini mencakup analisis data eksploratori, preprocessing, modeling, evaluasi, dan visualisasi.

## Struktur Proyek
tugas_decision_tree/
│
├── data/
│   └── titanic.csv
│
├── notebooks/
│   └── main.ipynb (file Jupyter Notebook utama)
│
├── src/ (folder untuk kode Python modular)
│   ├── __init__.py (file kosong untuk membuat folder sebagai package)
│   ├── preprocessing.py (berisi fungsi preprocessing)
│   ├── model.py (berisi fungsi modeling)
│   ├── visualization.py (berisi fungsi visualisasi)
│   └── utils.py (fungsi helper)
│
├── reports/
│   └── laporan.pdf
│
├── requirements.txt (list library yang diperlukan)
└── README.md (dokumentasi proyek) 

## Hasil
📝 Analisis Hasil
Insights Penting
Fitur Paling Penting: Jenis kelamin (Sex) adalah prediktor terkuat untuk survival

Optimal Depth: Model dengan max_depth=3 memberikan balance terbaik antara kompleksitas dan performa

Class Imbalance: Dataset memiliki distribusi yang relatif seimbang (38% survive)

Rekomendasi Improvement
Ensemble Methods: Coba Random Forest atau Gradient Boosting

Feature Engineering: Eksplorasi fitur baru seperti Title dari Name

Cross-Validation: Gunakan k-fold cross validation untuk validasi lebih robust

Hyperparameter Tuning: Eksplorasi parameter lain seperti min_samples_split

🎓 Implementasi Teori
Konsep Decision Tree yang Diimplementasikan
Node & Splitting: Pembagian data berdasarkan kondisi

Gini Impurity: Kriteria splitting yang digunakan

Pruning: Pembatasan max_depth untuk hindari overfitting

Feature Importance: Interpretasi kontribusi fitur

Perbandingan Metode
Decision Tree: Interpretable, mudah diimplementasi

Random Forest: Ensemble, lebih robust

Gradient Boosting: Sequential, high performance

## Cara Menjalankan
1. Install requirements: `pip install -r requirements.txt`
2. Run notebook: `jupyter notebook main.ipynb`