# 🧠 Smart Resume Analyzer

An **AI-powered web app** that analyzes how well a resume matches a given job description. It calculates a **match score (%)**, highlights **missing keywords**, and helps users improve their resumes for better job alignment.

---

## 🌟 Features
- 📄 Upload any resume (PDF format)
- 🧩 Paste a job description
- 🤖 AI analyzes text similarity using TF-IDF and cosine similarity
- 🔍 Lists missing keywords from the job description
- 📊 Provides a clear match score (0–100%)

---

## 🧰 Tech Stack
- Python
- Gradio – for the web interface
- Scikit-learn – for text similarity (TF-IDF + cosine similarity)
- PyMuPDF (fitz) – for reading PDFs
- NumPy – for numerical computations

---

## 🚀 Getting Started

Clone this repository, install dependencies, and run the app:

```bash
git clone https://github.com/jacob-okoth/smart-resume-analyzer.git && cd smart-resume-analyzer && pip install -r requirements.txt && python app.py
