import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import re
matplotlib.use('Agg')  # Mengubah backend ke 'Agg'
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Memuat data dari file CSV
df = pd.read_csv('kuesioner.csv')

# Mengganti nama kolom target untuk menghapus karakter newline dan masalah spasi
df.rename(columns={'8. Diterima atau tidak?\n- 1 = diterima\n- 0 = tidak': 'Diterima'}, inplace=True)

# Melihat tipe data dan deskripsi singkat
print(df.info())
print(df.describe())

# Membersihkan data
df.drop_duplicates(inplace=True)

# Analisis Eksplorasi Data (EDA)
# Visualisasi distribusi data
plt.figure(figsize=(10, 6))
sns.countplot(x='Diterima', data=df)
plt.title('Distribusi Diterima atau Tidak')
plt.savefig('distribusi_diterima.png')  # Simpan grafik ke file
plt.close()

# Menghapus kolom non-numerik sebelum menghitung korelasi
numerical_df = df.drop(columns=['Timestamp', 'Nama Lengkap'])

# Visualisasi hubungan antar fitur
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi antara Fitur')
plt.savefig('korelasi_fitur.png')  # Simpan grafik ke file
plt.close()

# Analisis hubungan fitur dengan target
features = ['1. Berapa durasi persiapan anda untuk mengikuti UTBK? (hari)',
            '2. Berapa lama rata-rata anda belajar per-harinya? (jam)',
            '3. Berapa nilai praktek ujian anda (skala 0-100)',
            '4. Berapa skor simulasi UTBK yang pernah anda ikuti? (Skala 1-1000, isi 0 jika tidak ada/tidak pernah ikut)',
            '5. Berapa lama durasi tidur anda selama persiapan UTBK? (jam)',
            '6. Berapa jumlah pertemuan bimbel yang anda ikuti untuk persiapan UTBK? (Isi 0 jika tidak ada/tidak pernah ikut)',
            '7. Seberapa percaya diri anda dalam persiapan UTBK (1-10)']

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Diterima', y=feature, data=numerical_df)
    plt.title(f'Boxplot {feature} vs Diterima')
    safe_feature = sanitize_filename(feature)
    plt.savefig(f'boxplot_{safe_feature}.png')  # Simpan grafik ke file
    plt.close()

# Menghitung korelasi dan signifikansi
for feature in features:
    diterima = numerical_df[numerical_df['Diterima'] == 1][feature]
    tidak_diterima = numerical_df[numerical_df['Diterima'] == 0][feature]
    t_stat, p_val = ttest_ind(diterima, tidak_diterima)
    print(f'Test Statistik {feature}: t-stat={t_stat}, p-value={p_val}')

# Melatih model dan mengevaluasi kinerja
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Membagi data
X = numerical_df.drop(columns=['Diterima'])
y = numerical_df['Diterima']

# Menormalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Melatih model RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Mengevaluasi model
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Akurasi Random Forest:", rf_accuracy)

# Feature Importance
feature_importance = rf_model.feature_importances_
features_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya Fitur': feature_importance})
features_df = features_df.sort_values(by='Pentingnya Fitur', ascending=False)
print(features_df)

# Visualisasi pentingnya fitur
plt.figure(figsize=(10, 6))
sns.barplot(x='Pentingnya Fitur', y='Fitur', data=features_df)
plt.title('Pentingnya Fitur')
plt.savefig('pentingnya_fitur.png')  # Simpan grafik ke file
plt.close()

# Kesimpulan
print("Kesimpulan:")
print("Dari analisis dan model yang dilatih, fitur yang paling berpengaruh terhadap diterima atau tidak adalah:")
print(features_df.head())
