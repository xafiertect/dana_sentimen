import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import joblib
import re

# LOAD DATASET

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_ready.csv")

    # normalisasi nama kolom
    df.columns = df.columns.str.lower().str.strip()

    # pastikan kolom utama ada
    if "review_text_clean" not in df.columns:
        st.error("Kolom 'review_text_clean' tidak ditemukan. Pastikan file dari notebook sudah benar.")
        st.stop()

    if "sentimen" not in df.columns:
        st.error("Kolom 'sentimen' tidak ditemukan.")
        st.stop()

    return df

df = load_data()

# LOAD MODEL

@st.cache_resource
def load_model():
    model = joblib.load("svm_sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

svm_model, tfidf = load_model()

# TEXT CLEANING (HARUS SAMA SEPERTI TRAINING)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_svm(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    return svm_model.predict(vec)[0]

# HEADER

st.title("App Review Intelligence Dashboard")
st.caption("AI Insight untuk Product Manager & UX Research")

# KPI METRICS

total = len(df)
negatif = (df["sentimen"] == "negatif").sum()
positif = (df["sentimen"] == "positif").sum()
netral  = (df["sentimen"] == "netral").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Review", total)
col2.metric("Positif", positif)
col3.metric("Netral", netral)
col4.metric("Negatif", negatif)

# EARLY WARNING SYSTEM

neg_ratio = negatif / total * 100

if neg_ratio > 30:
    st.error(f"ALERT: Sentimen Negatif Tinggi ({neg_ratio:.2f}%)")
else:
    st.success(f"Kondisi Normal ({neg_ratio:.2f}% negatif)")

# DISTRIBUSI SENTIMEN

st.subheader("Distribusi Sentimen")

fig, ax = plt.subplots()
df["sentimen"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Distribusi Sentimen")
ax.set_xlabel("Sentimen")
ax.set_ylabel("Jumlah Review")
st.pyplot(fig)

# ROOT CAUSE ANALYSIS

st.subheader("Keluhan Dominan (Root Cause Analysis)")

negative_reviews = " ".join(
    df[df["sentimen"] == "negatif"]["review_text_clean"].astype(str)
)

words = negative_reviews.split()
word_freq = Counter(words).most_common(15)

word_df = pd.DataFrame(word_freq, columns=["Keyword", "Frekuensi"])
st.dataframe(word_df)

# SAMPLE REVIEW NEGATIF

st.subheader("Contoh Review Negatif")

sample_data = df[df["sentimen"] == "negatif"]["review_text_clean"]

if len(sample_data) > 10:
    st.dataframe(sample_data.sample(10))
else:
    st.dataframe(sample_data)

# PRODUCT QUESTION ENGINE

st.divider()
st.subheader("Tanya Insight Produk")

question = st.text_input("Contoh: Kenapa rating turun bulan ini?")

if question:

    st.write("### Analisis Sistem")

    if "turun" in question.lower() or "kenapa" in question.lower():

        top_issues = word_df.head(5)["Keyword"].tolist()

        st.warning(
            f"Penurunan kepuasan pengguna dipicu oleh isu utama: {', '.join(top_issues)}"
        )

        st.write("### Rekomendasi Action:")
        st.write("- Investigasi bug terkait keyword di atas")
        st.write("- Prioritaskan hotfix pada issue paling sering muncul")
        st.write("- Tingkatkan reliability sistem (login, OTP, transaksi)")
        st.write("- Monitor ulang sentimen setelah release patch")

    else:
        st.info("Pertanyaan dikenali sebagai analisis umum.")

# LIVE SENTIMENT TESTING

st.divider()
st.subheader("Analisis Review Baru (Realtime Model)")

text = st.text_area("Masukkan review pengguna:")

if st.button("Analisis Sentimen"):
    if text.strip():
        result = predict_svm(text)
        st.success(f"Hasil Prediksi: {result.upper()}")
    else:
        st.warning("Masukkan teks terlebih dahulu.")