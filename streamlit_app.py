import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import load_and_preprocess
from train_model import build_model
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Judul
st.title("📊 Prediksi Penjualan Produk Cetakan - Gramedia Printing")

# Upload CSV
uploaded_file = st.file_uploader("Unggah file CSV data penjualan", type=["csv"])

if uploaded_file:
    # Simpan ke sementara dan load
    df = pd.read_csv(uploaded_file)
    st.write("📋 Data yang diunggah:")
    st.dataframe(df)

    # Simpan sementara
    df.to_csv("data/temp_upload.csv", index=False)

    # Proses
    try:
        X_scaled, y = load_and_preprocess("data/temp_upload.csv")
        model = load_model("model/ann_model.h5")

        # Prediksi
        y_pred = model.predict(X_scaled).flatten()

        # Evaluasi
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        st.success(f"📈 Evaluasi Model:\n\n- MAE: {mae:.2f}\n- RMSE: {rmse:.2f}\n- R²: {r2:.2f}")

        # Plot
        st.subheader("📊 Grafik Aktual vs Prediksi")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y, y=y_pred, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        ax.set_xlabel("Aktual")
        ax.set_ylabel("Prediksi")
        ax.set_title("Prediksi vs Aktual Penjualan")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file CSV untuk mulai prediksi.")
