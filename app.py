import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 0. KONFIGURASI HALAMAN STREAMLIT (HARUS JADI YANG PERTAMA!) ---
st.set_page_config(
    page_title="Prediktor Spesies Penguin",
    page_icon="🐧",
    layout="centered"
)

# --- 1. Muat Model dan Objek Pra-pemrosesan yang Telah Disimpan ---
# Pastikan file-file .pkl ada di direktori yang sama
try:
    model = joblib.load('penguin_classifier_model.pkl')
    scaler = joblib.load('penguin_scaler.pkl')
    le_species = joblib.load('penguin_le_species.pkl')
    le_sex = joblib.load('penguin_le_sex.pkl')
    st.success("Model dan objek pra-pemrosesan berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: Salah satu file model/scaler/encoder tidak ditemukan. Pastikan Anda telah menjalankan skrip ML untuk menyimpannya.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan

# --- 2. Konten Utama Aplikasi ---
st.title("🐧 Prediktor Spesies Penguin")
st.markdown("Aplikasi ini memprediksi spesies penguin berdasarkan pengukuran tubuh.")
st.markdown("Masukkan karakteristik penguin di bawah ini:")

# --- 3. Input Pengguna (Fitur) ---
st.header("Parameter Pengukuran:")

# Slider untuk Culmen Length (mm)
culmen_length = st.slider(
    "Panjang Paruh (Culmen Length) (mm)",
    min_value=30.0,
    max_value=60.0,
    value=45.0,
    step=0.1,
    help="Panjang paruh penguin dari pangkal hingga ujung."
)

# Slider untuk Culmen Depth (mm)
culmen_depth = st.slider(
    "Kedalaman Paruh (Culmen Depth) (mm)",
    min_value=13.0,
    max_value=22.0,
    value=17.0,
    step=0.1,
    help="Kedalaman paruh penguin di bagian terlebar."
)

# Slider untuk Flipper Length (mm)
flipper_length = st.slider(
    "Panjang Sirip (Flipper Length) (mm)",
    min_value=170.0,
    max_value=240.0,
    value=200.0,
    step=1.0,
    help="Panjang sirip penguin."
)

# Slider untuk Body Mass (g)
body_mass = st.slider(
    "Massa Tubuh (Body Mass) (g)",
    min_value=2500.0,
    max_value=6500.0,
    value=4000.0,
    step=50.0,
    help="Massa tubuh penguin."
)

# Pilihan untuk Sex
sex_options = le_sex.classes_ # Ambil kelas dari le_sex yang sudah di-fit
sex = st.selectbox(
    "Jenis Kelamin (Sex)",
    options=sex_options,
    index=0, # Default ke opsi pertama (biasanya 'FEMALE' atau 'MALE' tergantung urutan)
    help="Jenis kelamin penguin."
)

# --- 4. Tombol Prediksi ---
if st.button("Prediksi Spesies Penguin"):
    # --- 5. Pra-pemrosesan Input Pengguna ---
    # Encode 'sex' yang dipilih pengguna
    sex_encoded = le_sex.transform([sex])[0]

    # Buat array NumPy dari input pengguna
    # PASTIKAN URUTAN KOLOM SAMA dengan X_train saat melatih model:
    # ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex_encoded']
    input_data = np.array([[culmen_length, culmen_depth, flipper_length, body_mass, sex_encoded]])

    # Scaling input pengguna menggunakan scaler yang SAMA
    input_data_scaled = scaler.transform(input_data)

    # --- 6. Melakukan Prediksi ---
    predicted_species_encoded = model.predict(input_data_scaled)
    predicted_species_name = le_species.inverse_transform(predicted_species_encoded)

    # Dapatkan probabilitas (kepercayaan model)
    prediction_probabilities = model.predict_proba(input_data_scaled)
    predicted_probability = np.max(prediction_probabilities) * 100

    # --- 7. Tampilkan Hasil Prediksi ---
    st.subheader("Hasil Prediksi:")
    st.success(f"Berdasarkan input, penguin ini kemungkinan besar adalah **{predicted_species_name[0]}**.")
    st.info(f"Model memprediksi ini dengan kepercayaan sekitar **{predicted_probability:.2f}%**.")

    st.write("---")
    st.subheader("Detail Input:")
    st.write(f"Panjang Paruh: {culmen_length} mm")
    st.write(f"Kedalaman Paruh: {culmen_depth} mm")
    st.write(f"Panjang Sirip: {flipper_length} mm")
    st.write(f"Massa Tubuh: {body_mass} g")
    st.write(f"Jenis Kelamin: {sex}")

# --- Bagian "About" & Footer ---
st.write("---") # Garis pemisah
st.header("📚 Tentang Dataset Penguin")
st.markdown(
    """
    Dataset ini berisi pengukuran morfologis untuk tiga spesies penguin yang berbeda:
    * **Adelie Penguin**
    * **Chinstrap Penguin**
    * **Gentoo Penguin**

    Data ini dikumpulkan dari Kepulauan Palmer, Antartika. Fitur-fitur utama yang digunakan untuk prediksi meliputi:
    * **Panjang Paruh (Culmen Length)**: Panjang paruh penguin dari pangkal hingga ujung.
    * **Kedalaman Paruh (Culmen Depth)**: Kedalaman paruh penguin di bagian terlebar.
    * **Panjang Sirip (Flipper Length)**: Panjang sirip penguin.
    * **Massa Tubuh (Body Mass)**: Berat tubuh penguin.
    * **Jenis Kelamin (Sex)**: Jenis kelamin penguin (jantan/betina).

    Dataset ini sangat populer di dunia Machine Learning untuk tugas **klasifikasi multi-kelas** karena pola yang jelas antar spesies, menjadikannya ideal untuk pembelajaran dan demonstrasi.
    """
)

st.write("---") # Garis pemisah
st.markdown("Dibuat oleh: **Wahyu Andika Rahadi**")
st.caption("Aplikasi ini dibuat untuk tujuan demonstrasi dan pembelajaran Machine Learning.")