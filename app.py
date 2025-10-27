import gradio as gr
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 🔹 Function: Extract text from a PDF resume
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    try:
        # Handle both file path and uploaded file objects
        if hasattr(pdf_file, "name"):
            path = pdf_file.name  # for Gradio NamedString
        else:
            path = pdf_file  # fallback for file paths

        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# ---------------------------------------------------------
# 🔹 Function: Compute similarity between resume & job description
# ---------------------------------------------------------
def match_resume_to_job(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

# ---------------------------------------------------------
# 🔹 Main Gradio interface function
# ---------------------------------------------------------
def analyze_resume(pdf, job_description):
    try:
        resume_text = extract_text_from_pdf(pdf)
        if "Error" in resume_text:
            return f"❌ {resume_text}"

        score = match_resume_to_job(resume_text, job_description)
        feedback = f"✅ Resume matches the job description by **{score}%**"
        if score < 50:
            feedback += "\n💡 Try adding more relevant keywords or skills."
        elif score < 75:
            feedback += "\n⚙️ Good match, but you can tailor your resume more."
        else:
            feedback += "\n🔥 Excellent! Your resume aligns very well."
        return feedback
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# 🔹 Gradio UI
# ---------------------------------------------------------
interface = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="📄 Upload Your Resume (PDF)"),
        gr.Textbox(label="💼 Paste Job Description Here")
    ],
    outputs="markdown",
    title="Smart Resume Analyzer",
    description="Upload your resume and paste a job description to see how well they match."
)

# ---------------------------------------------------------
# 🔹 Launch App
# ---------------------------------------------------------
if __name__ == "__main__":
    interface.launch()
