import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Cache models so they don’t reload every time
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return summarizer, classifier

summarizer, classifier = load_models()

# TF-IDF keyword extractor
def extract_keywords_tfidf(text, top_n=8, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = X.toarray()[0]
    top_indices = scores.argsort()[::-1][:top_n]
    return [(feature_names[i], round(float(scores[i]), 4)) for i in top_indices]

# Streamlit UI
st.set_page_config(page_title="Astronomy Hub Analyzer", layout="wide")
st.title("🔭 Astronomy Article Analyzer")
st.write("Paste an astronomy article and get a **summary, keywords, and classification** powered by Deep Learning.")

article_text = st.text_area("Paste your astronomy article here:", height=200)

if st.button("Analyze"):
    if article_text.strip():
        # Summarization
        with st.spinner("Generating summary..."):
            summary = summarizer(article_text, max_length=60, min_length=25, do_sample=False)[0]['summary_text']
        
        # Keywords
        with st.spinner("Extracting keywords..."):
            keywords = extract_keywords_tfidf(article_text, top_n=8)
        
        # Classification
        with st.spinner("Classifying article..."):
            candidate_labels = ["Exoplanets", "Galaxies", "Stars and Nebulae", "Space Missions", "Astronomy General"]
            classification = classifier(article_text, candidate_labels)

        # Show results
        st.subheader("✍️ Summary")
        st.success(summary)

        st.subheader("🗝️ Top Keywords")
        for kw, score in keywords:
            st.write(f"- {kw} (score: {score})")

        st.subheader("🚀 Classification")
        for label, score in zip(classification['labels'], classification['scores']):
            st.write(f"- {label}: {round(score, 3)}")

    else:
        st.warning("⚠️ Please paste some text first.")
