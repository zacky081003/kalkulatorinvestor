import streamlit as st
import pandas as pd
import numpy as np
import bcrypt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO












st.set_page_config(page_title="Dashboard Klasterisasi Saham Indonesia ",
                   layout="wide")

st.title("📊 Dashboard Sistem Informasi Klasterisasi Saham Indonesia")
st.write("Analisis klasterisasi saham berdasarkan rasio keuangan ROA, ROE, DER dan PER menggunakan algoritma **K-Means**.")

st.markdown("---")

# Upload file
file = st.file_uploader("📎 Upload File Excel Saham", type=["xlsx"])

if file is None:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

data = pd.read_excel(file)
data.columns = data.columns.str.strip()

st.success("Data berhasil dimuat!")
st.dataframe(data, use_container_width=True)

# Variabel
fitur = ["ROA", "ROE", "DER", "PER"]
fitur = [f for f in fitur if f in data.columns]

if len(fitur) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom dari ROA, ROE, DER, PER")
    st.stop()

X = data[fitur]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)




## =========================
# PILIH FITUR
# =========================
st.subheader(" Pilih Variabel Rasio Keuangan")

default_cols = [c for c in ["ROA","ROE","DER","PER"] if c in data.columns]

fitur = st.multiselect(
    "Pilih kolom numerik:",
    options=list(data.columns),
    default=default_cols
)

if len(fitur)==0:
    st.error("Pilih minimal satu kolom numerik.")
    st.stop()

df = data[fitur].copy()
df = df.dropna()

# =========================
# PREPROCESSING
# =========================
st.subheader(" Pre-Processing Data")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

st.write("### 🔹 Data Setelah Normalisasi (Min-Max)")
st.dataframe(pd.DataFrame(scaled, columns=fitur), use_container_width=True)

# ======================================================
# 🔹 Elbow Method
# ======================================================
st.header("📉 Evaluasi Elbow Method (Menentukan K Terbaik)")

inertia = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(K_range, inertia, marker='o')
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("Nilai Inertia")
ax.set_title("Grafik Elbow Method")
st.pyplot(fig)

st.info("🔎 Titik siku (elbow) biasanya menunjukkan jumlah cluster yang optimal.")

# ======================================================
# 🔹 Pilih K
# ======================================================
st.header("🟢 Tentukan Jumlah Cluster (K)")
k = st.slider("Pilih jumlah cluster:", 2, 6, 3)

model = KMeans(n_clusters=k, random_state=42)
clusters = model.fit_predict(X_scaled)

data["Cluster"] = clusters

st.success("Proses clustering berhasil dilakukan!")

# ======================================================
# 🔹 Visualisasi Cluster (ROA vs ROE)
# ======================================================
st.header("🟣 Visualisasi Hasil Klasterisasi Saham ")

fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(
    data=data,
    x="ROA",
    y="ROE",
    hue="Cluster",
    palette="Set2",
    s=120,
    ax=ax
)

ax.set_title("Visualisasi K-Means Clustering (ROA sebagai X, ROE sebagai Y)")
ax.set_xlabel("Return on Assets (ROA)")
ax.set_ylabel("Return on Equity (ROE)")
ax.legend(title="Cluster")

st.pyplot(fig)

st.info(
    "Grafik ini menampilkan hasil klasterisasi saham menggunakan algoritma K-Means. "
    "Sumbu X menunjukkan ROA dan sumbu Y menunjukkan ROE, "
    "sedangkan pembentukan cluster tetap mempertimbangkan ROA, ROE, DER, dan PER."
)



# ======================================================
# 🔹 Hasil Klasterisasi
# ======================================================
st.header("📊 Hasil Klasterisasi Saham")
st.dataframe(data, use_container_width=True)


# ======================================================
# 🔹 Dashboard Analitik
# ======================================================
st.header("🧠  Analitik Data")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rata-rata ROA", f"{data['ROA'].mean():.2f}")
col2.metric("Rata-rata ROE", f"{data['ROE'].mean():.2f}")
col3.metric("Rata-rata DER", f"{data['DER'].mean():.2f}")
col4.metric("Rata-rata PER", f"{data['PER'].mean():.2f}")

st.markdown("### 📌 Jumlah Perusahaan per Cluster")
st.bar_chart(data["Cluster"].value_counts().sort_index())

# ======================================================
# 🔹 Rata-rata Rasio per Cluster
# ======================================================
st.header("📈 Rata-Rata Rasio Keuangan per Cluster")

summary = data.groupby("Cluster")[fitur].mean()

st.dataframe(summary, use_container_width=True)

fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(summary.reset_index().melt(id_vars="Cluster"), x="Cluster", y="value", hue="variable")
plt.title("Grafik Rata-Rata Rasio Keuangan per Cluster")
st.pyplot(fig)

# ======================================================
# 🔹 Interpretasi Otomatis + Grafik Bar Cluster (Fix)
# ======================================================
st.header("🧾 Interpretasi Otomatis Per Cluster")

# pastikan summary dihitung
summary = data.groupby("Cluster")[fitur].mean()

# rata-rata global untuk pembanding
mean_vals = summary.mean()

# loop hanya cluster yang ADA
for i in sorted(data["Cluster"].unique()):

    st.subheader(f"🔹 Cluster {i}")
    subset = data[data["Cluster"] == i]

    st.write(f"Berisi **{len(subset)} perusahaan:**")
    st.dataframe(subset[["Kode","Perusahaan"] + fitur], use_container_width=True)

    # ======================
    # GRAFIK RATA-RATA CLUSTER
    # ======================
    fig, ax = plt.subplots(figsize=(6,3.5))
    summary.loc[i].plot(kind="bar", ax=ax)
    ax.set_title(f"Rata-rata Rasio Keuangan Cluster {i}")
    ax.tick_params(axis='both', labelsize=8)
    st.pyplot(fig)

    # ======================
    # LOGIKA PENILAIAN CLUSTER
    # ======================
    score = 0
    note = []

    if "ROA" in summary.columns:
        if summary.loc[i,"ROA"] >= mean_vals["ROA"]:
            score += 1
            note.append("✔ ROA tinggi (profitabilitas baik)")
        else:
            note.append("⚠ ROA rendah")

    if "ROE" in summary.columns:
        if summary.loc[i,"ROE"] >= mean_vals["ROE"]:
            score += 1
            note.append("✔ ROE tinggi (return investor baik)")
        else:
            note.append("⚠ ROE rendah")

    if "DER" in summary.columns:
        if summary.loc[i,"DER"] <= mean_vals["DER"]:
            score += 1
            note.append("✔ DER rendah (risiko hutang kecil)")
        else:
            note.append("⚠ DER tinggi (risiko hutang besar)")

    # ======================
    # KATEGORI FINAL
    # ======================
    if score == 3:
        kategori = "🟢 **Cluster Baik (Performa Tinggi)**"
    elif score == 2:
        kategori = "🟡 **Cluster Sedang (Stabil / Moderat)**"
    else:
        kategori = "🔴 **Cluster Kurang Baik (Risiko Tinggi / Profit Rendah)**"

    st.markdown(f"### {kategori}")
    st.write("\n".join(note))

# ======================================================
# 🔹 Download Excel
# ======================================================
buffer = BytesIO()
data.to_excel(buffer, index=False)
buffer.seek(0)

st.download_button("📥 Download Hasil Klasterisasi (.xlsx)",
                   data=buffer,
                   file_name="hasil_cluster.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("📘 Aplikasi ini dikembangkan untuk Tugas Akhir D3 Sistem Informasi — © Zacky Oktaviansyah")

