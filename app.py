"""
Aplikasi Streamlit — Prediksi Segmen Nasabah Kartu Kredit (Customer Segmentation)
Proyek: Implementasi Clustering untuk Menemukan Pola pada Data (Metodologi CRISP-DM)
Dataset: Credit Card customers (BankChurners.csv) - Kaggle, oleh Sakshi Goyal

Cara menjalankan lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit - K-Means Clustering",
    page_icon="💳",
    layout="centered",
)

FITUR_CLUSTER = ["Customer_Age", "Credit_Limit", "Total_Revolving_Bal",
                  "Total_Trans_Amt", "Total_Trans_Ct", "Avg_Utilization_Ratio"]

PERSONA = {
    0: {
        "nama": "Revolver Berisiko",
        "deskripsi": "Batas kredit relatif rendah namun saldo revolving yang dibawa tinggi — rasio utilisasi kartu paling tinggi (~53%).",
        "rekomendasi": "Pantau risiko kredit secara berkala, tawarkan skema cicilan/restrukturisasi, berikan edukasi keuangan agar tidak menunggak.",
    },
    1: {
        "nama": "Nasabah Sehat / Transactors",
        "deskripsi": "Jarang membawa saldo dari bulan ke bulan (utilisasi kartu paling rendah, ~5%) — kemungkinan besar rutin melunasi tagihan.",
        "rekomendasi": "Pertahankan dengan program reward/loyalitas, tawarkan cross-sell produk simpanan atau investasi.",
    },
    2: {
        "nama": "Nasabah Premium / Elite",
        "deskripsi": "Batas kredit jauh di atas nasabah lain, namun aktivitas transaksi masih di level standar.",
        "rekomendasi": "Tawarkan layanan prioritas, kartu premium, dan penawaran eksklusif untuk mendorong pemakaian lebih tinggi.",
    },
    3: {
        "nama": "Power User / Heavy Spender",
        "deskripsi": "Frekuensi dan nilai transaksi jauh di atas kelompok lain — nasabah paling aktif menggunakan kartu.",
        "rekomendasi": "Berikan program cashback/rewards proporsional dengan pengeluaran, prioritaskan retensi nasabah ini.",
    },
}


@st.cache_resource
def muat_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data
def muat_profil():
    return pd.read_csv("cluster_profile.csv")


def main():
    st.title("💳 Prediksi Segmen Nasabah Kartu Kredit")
    st.caption("Berdasarkan model K-Means Clustering — proyek Data Science CRISP-DM")

    st.markdown(
        "Masukkan data profil & aktivitas kartu kredit nasabah pada form di bawah untuk "
        "memprediksi segmen (cluster) nasabah tersebut, lengkap dengan interpretasi bisnisnya."
    )

    try:
        model, scaler = muat_model()
        profil_df = muat_profil()
    except FileNotFoundError:
        st.error(
            "File model (kmeans_model.pkl / scaler.pkl / cluster_profile.csv) belum ditemukan. "
            "Pastikan seluruh file tersebut berada pada folder yang sama dengan app.py."
        )
        return

    with st.form("form_nasabah"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia nasabah (Customer_Age)", min_value=18, max_value=90, value=45)
            credit_limit = st.number_input("Batas limit kartu (Credit_Limit)", min_value=0.0, max_value=40000.0, value=5000.0, step=500.0)
            revolving_bal = st.number_input("Saldo revolving (Total_Revolving_Bal)", min_value=0.0, max_value=3000.0, value=800.0, step=50.0)
        with col2:
            trans_amt = st.number_input("Total nilai transaksi (Total_Trans_Amt)", min_value=0.0, max_value=20000.0, value=3500.0, step=100.0)
            trans_ct = st.number_input("Total jumlah transaksi (Total_Trans_Ct)", min_value=0, max_value=150, value=60)
            utilization = st.slider("Rasio utilisasi kartu (Avg_Utilization_Ratio)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        submitted = st.form_submit_button("Prediksi Segmen")

    if submitted:
        data_baru = pd.DataFrame(
            [[age, credit_limit, revolving_bal, trans_amt, trans_ct, utilization]],
            columns=FITUR_CLUSTER,
        )

        data_scaled = scaler.transform(data_baru)
        cluster_pred = int(model.predict(data_scaled)[0])

        persona = PERSONA.get(cluster_pred, {"nama": f"Cluster {cluster_pred}", "deskripsi": "-", "rekomendasi": "-"})

        st.success(f"Nasabah ini termasuk **Cluster {cluster_pred}: {persona['nama']}**")
        st.write(f"**Karakteristik:** {persona['deskripsi']}")
        st.write(f"**Rekomendasi strategi:** {persona['rekomendasi']}")

        st.markdown("---")
        st.subheader("Perbandingan dengan Profil Rata-rata Tiap Cluster")
        st.dataframe(profil_df.set_index("Cluster"), use_container_width=True)

    with st.expander("Tentang Proyek Ini"):
        st.markdown(
            """
            Aplikasi ini merupakan tahap **Deployment** dari proyek clustering nasabah
            kartu kredit yang mengikuti metodologi **CRISP-DM**. Model K-Means dilatih pada
            dataset **Credit Card customers (BankChurners.csv)** dari Kaggle (oleh Sakshi Goyal),
            berisi data 10.127 nasabah kartu kredit.

            Catatan: dataset ini aslinya dibuat untuk kasus prediksi churn nasabah, namun pada
            proyek ini digunakan murni untuk clustering/segmentasi perilaku (kolom churn tidak dipakai).

            Notebook lengkap proses Business Understanding hingga Evaluation tersedia
            pada file `clustering_cc_crispdm.ipynb` pada repository proyek ini.
            """
        )


if __name__ == "__main__":
    main()