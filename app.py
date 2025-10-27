import gradio as gr
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def keyword_match(job_text, resume_text):
    job_keywords = set(re.findall(r'\b[A-Za-z]+\b', job_text.lower()))
    resume_keywords = set(re.findall(r'\b[A-Za-z]+\b', resume_text.lower()))
    missing = job_keywords - resume_keywords
    return list(missing)[:10]

def analyze_resume(resume_file, job_description):
    resume_text = extract_text_from_pdf(resume_file)
    score = calculate_similarity(resume_text, job_description)
    missing = keyword_match(job_description, resume_text)
    return {
        "Match Score (%)": score,
        "Missing Keywords": ", ".join(missing) if missing else "None! Great fit."
    }

demo = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF)"),
        gr.Textbox(label="Paste Job Description", lines=10)
    ],
    outputs=[gr.JSON(label="Analysis Result")],
    title="Smart Resume Analyzer",
    description="Upload your resume and see how well it matches a job description."
)

if __name__ == "__main__":
    demo.launch()
