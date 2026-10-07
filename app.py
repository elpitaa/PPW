"""Streamlit app for classifying a Detik article with an existing .pkl model."""

import pickle
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup


# Ganti nama file ini jika model .pkl kamu memiliki nama berbeda.
MODEL_PATH = Path(__file__).with_name("model_berita.pkl")
MIN_CONFIDENCE = 0.60
SUPPORTED_DOMAINS = {"finance.detik.com", "sport.detik.com"}


def clean_text(text):
    text = "".join(
        character
        for character in str(text)
        if not unicodedata.category(character).startswith("P")
    )
    return re.sub(r"\s+", " ", text).strip()


def fetch_article(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Masukkan URL lengkap yang diawali http:// atau https://.")
    hostname = (parsed.hostname or "").lower()
    if hostname not in SUPPORTED_DOMAINS:
        raise ValueError(
            "Berita ini berada di luar cakupan model. "
            "Masukkan berita dari finance.detik.com atau sport.detik.com."
        )

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title_node = soup.select_one("h1.detail__title, h1")
    body_nodes = soup.select("div.detail__body-text") or soup.select("article p")
    title = title_node.get_text(" ", strip=True) if title_node else ""
    body = " ".join(node.get_text(" ", strip=True) for node in body_nodes)
    if not body:
        raise ValueError("Isi berita tidak ditemukan dari URL tersebut.")
    return title, body


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File model_berita.pkl belum ada. Letakkan file .pkl di folder yang sama dengan app.py."
        )
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def predict(text, bundle):
    tokens = clean_text(text).split()
    word_to_index = bundle["word_to_index"]
    embeddings = np.asarray(bundle["word_embeddings"])
    known_vectors = [
        embeddings[word_to_index[word]]
        for word in tokens
        if word in word_to_index
    ]
    vector_size = bundle["vector_size"]
    document_vector = (
        np.mean(known_vectors, axis=0)
        if known_vectors
        else np.zeros(vector_size)
    )
    classifier = bundle["classifier"]
    prediction = int(classifier.predict([document_vector])[0])
    probabilities = classifier.predict_proba([document_vector])[0]
    labels = bundle["label_mapping"]
    scores = {
        labels[int(class_index)]: float(probability)
        for class_index, probability in zip(classifier.classes_, probabilities)
    }
    return labels[prediction], scores


st.set_page_config(page_title="Klasifikasi Berita Detik", page_icon="📰")
st.title("Klasifikasi Berita Detik")
st.write("Masukkan link berita Detik untuk memprediksi kelas finance atau sport.")

url = st.text_input("Link berita Detik", placeholder="https://sport.detik.com/...")

if st.button("Klasifikasikan", type="primary"):
    if not url.strip():
        st.warning("Masukkan link berita terlebih dahulu.")
    else:
        try:
            with st.spinner("Mengambil berita dan melakukan prediksi..."):
                title, body = fetch_article(url.strip())
                result, scores = predict(title + " " + body, load_model())
            st.subheader(title or "Berita Detik")
            st.write("Probabilitas:")
            st.write(f"Finance: {scores.get('finance', 0) * 100:.2f}%")
            st.write(f"Sport: {scores.get('sport', 0) * 100:.2f}%")
            confidence = scores.get(result, 0.0)
            if confidence < MIN_CONFIDENCE:
                st.warning(
                    "Berita ini belum dapat dikategorikan dengan yakin sebagai "
                    "finance atau sport."
                )
                st.caption(
                    f"Keyakinan tertinggi hanya {confidence * 100:.2f}% "
                    f"(batas: {MIN_CONFIDENCE * 100:.0f}%)."
                )
            else:
                st.success(f"Hasil klasifikasi: {result.upper()}")
        except (requests.RequestException, FileNotFoundError, ValueError) as error:
            st.error(str(error))
