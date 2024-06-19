import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Memuat data dari file CSV
df = pd.read_csv('kuesioner.csv')

# Mengganti nama kolom target untuk menghapus karakter newline dan masalah spasi
df.rename(columns={'8. Diterima atau tidak?\n- 1 = diterima\n- 0 = tidak': 'Diterima'}, inplace=True)

# Membagi data
X = df.drop(columns=['Diterima', 'Timestamp', 'Nama Lengkap'])
y = df['Diterima']

# Menormalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Melatih model RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Melatih model Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Melatih model K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Menerapkan K-Means clustering
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X_train)
kmeans_pred = kmeans_model.predict(X_test)

# Melatih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Mengevaluasi model
rf_pred = rf_model.predict(X_test)
lr_pred = np.round(lr_model.predict(X_test)).astype(int)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Mengonversi prediksi kmeans untuk mencocokkan label asli (0 dan 1)
cluster_to_label = {i: 1 if sum(y_train[kmeans_model.labels_ == i]) > len(y_train[kmeans_model.labels_ == i]) / 2 else 0 for i in range(2)}
kmeans_pred = np.array([cluster_to_label[label] for label in kmeans_pred])

# Menghitung akurasi
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
kmeans_accuracy = accuracy_score(y_test, kmeans_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Akurasi Random Forest:", rf_accuracy)
print("Akurasi Linear Regression:", lr_accuracy)
print("Akurasi K-Nearest Neighbors:", knn_accuracy)
print("Akurasi K-Means Clustering:", kmeans_accuracy)
print("Akurasi Decision Tree:", dt_accuracy)

# Membandingkan akurasi dan menemukan model terbaik
akurasi = {
    'Random Forest': rf_accuracy,
    'Linear Regression': lr_accuracy,
    'K-Nearest Neighbors': knn_accuracy,
    'K-Means Clustering': kmeans_accuracy,
    'Decision Tree': dt_accuracy
}

model_terbaik = max(akurasi, key=akurasi.get)
print("Model terbaik adalah:", model_terbaik, "dengan akurasi:", akurasi[model_terbaik])

# Membuat data dummy untuk prediksi
dummy_data = {
    '1. Berapa durasi persiapan anda untuk mengikuti UTBK? (hari)': np.random.randint(1, 366, 20),
    '2. Berapa lama rata-rata anda belajar per-harinya? (jam)': np.random.randint(1, 10, 20),
    '3. Berapa nilai praktek ujian anda (skala 0-100)': np.random.randint(0, 101, 20),
    '4. Berapa skor simulasi UTBK yang pernah anda ikuti? (Skala 1-1000, isi 0 jika tidak ada/tidak pernah ikut)': np.random.randint(0, 1001, 20),
    '5. Berapa lama durasi tidur anda selama persiapan UTBK? (jam)': np.random.randint(1, 13, 20),
    '6. Berapa jumlah pertemuan bimbel yang anda ikuti untuk persiapan UTBK? (Isi 0 jika tidak ada/tidak pernah ikut)': np.random.randint(0, 61, 20),
    '7. Seberapa percaya diri anda dalam persiapan UTBK (1-10)': np.random.randint(1, 11, 20)
}

dummy_df = pd.DataFrame(dummy_data)
dummy_df.to_csv('dummy_data.csv', index=False)

# Menormalisasi data dummy
dummy_scaled = scaler.transform(dummy_df)

# Membuat prediksi menggunakan model terbaik
if model_terbaik == 'Random Forest':
    dummy_pred = rf_model.predict(dummy_scaled)
elif model_terbaik == 'Linear Regression':
    dummy_pred = np.round(lr_model.predict(dummy_scaled)).astype(int)
elif model_terbaik == 'K-Nearest Neighbors':
    dummy_pred = knn_model.predict(dummy_scaled)
elif model_terbaik == 'K-Means Clustering':
    dummy_kmeans_pred = kmeans_model.predict(dummy_scaled)
    dummy_pred = np.array([cluster_to_label[label] for label in dummy_kmeans_pred])
else:
    dummy_pred = dt_model.predict(dummy_scaled)

dummy_df['Diterima'] = dummy_pred
print("Prediksi untuk data dummy:")
print(dummy_df)
