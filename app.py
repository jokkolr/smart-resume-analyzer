import gradio as gr
import PyMuPDF  # Correct library import
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------
# 🔹 Function: Extract text from uploaded PDF
# -------------------------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as f:
        pdf = PyMuPDF.open(f)
        for page in pdf:
            text += page.get_text("text")
    return text


# -------------------------------------------
# 🔹 Function: Analyze resume vs job description
# -------------------------------------------
def analyze_resume(resume_pdf, job_description):
    try:
        # Extract text from the uploaded resume
        resume_text = extract_text_from_pdf(resume_pdf.name)

        # Combine resume and job description for vectorization
        texts = [resume_text, job_description]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Calculate similarity (cosine similarity)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

        # Identify missing keywords
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        missing_keywords = job_words - resume_words
        missing_keywords = ", ".join(list(missing_keywords)[:10])  # limit to 10

        result = f"✅ **Match Score:** {similarity:.2f}%\n\n❌ **Missing Keywords:** {missing_keywords if missing_keywords else 'None'}"
        return result

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# -------------------------------------------
# 🔹 Gradio Interface
# -------------------------------------------
title = "🧠 Smart Resume Analyzer"
description = "Upload your resume (PDF) and paste a job description to see how well they match!"

interface = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF)"),
        gr.Textbox(label="Paste Job Description", lines=8, placeholder="Paste job description here..."),
    ],
    outputs=gr.Markdown(label="Results"),
    title=title,
    description=description,
)

# -------------------------------------------
# 🔹 Launch the App
# -------------------------------------------
if __name__ == "__main__":
    interface.launch(share=True)
