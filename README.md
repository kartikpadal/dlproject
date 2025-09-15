# 🔭 Astronomy Article Analyzer

A Streamlit web app that:
- Summarizes astronomy articles (BART model)
- Extracts top keywords (TF-IDF)
- Classifies articles into categories (BART-MNLI)

## Run Locally
```bash
pip install streamlit transformers torch scikit-learn
OR
python -m streamlit run app.py
