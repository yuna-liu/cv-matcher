import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# === Streamlit UI ===
st.title("🎯 Job Matcher (Multiple CVs)")

st.write("""
Upload up to 5 CV PDFs and paste the job description.
The AI will compare each CV with the job description and return a ranking, similarity scores, 
and suggestions for missing skills or experience.
""")

# --- Upload CVs ---
uploaded_cvs = st.file_uploader(
    "Upload CV PDFs (max 5)", 
    type="pdf", 
    accept_multiple_files=True
)

# --- Job Description ---
job_description = st.text_area(
    "Paste the Job Description here",
    placeholder="Job title, responsibilities, requirements..."
)

# --- Hugging Face Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def pdf_to_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

if uploaded_cvs and job_description:
    if len(uploaded_cvs) > 5:
        st.warning("Please upload up to 5 CVs only.")
    else:
        # --- Convert PDFs to text ---
        cv_texts = [pdf_to_text(cv) for cv in uploaded_cvs]
        cv_names = [cv.name for cv in uploaded_cvs]

        # --- Compute embeddings ---
        cv_embeddings = model.encode(cv_texts)
        job_embedding = model.encode([job_description])

        # --- Compute cosine similarity ---
        similarities = cosine_similarity(cv_embeddings, job_embedding).flatten()

        # --- Build results dataframe ---
        results = pd.DataFrame({
            "CV Name": cv_names,
            "Match Score": (similarities * 100).round(2)
        }).sort_values(by="Match Score", ascending=False).reset_index(drop=True)

        st.subheader("📊 CV Ranking")
        st.dataframe(results)

        # --- Optional suggestions ---
        st.subheader("💡 Suggestions / Highlights")
        for i, score in enumerate(similarities):
            if score < 0.6:
                st.markdown(f"- **{cv_names[i]}**: Consider highlighting more relevant skills or experience.")
            else:
                st.markdown(f"- **{cv_names[i]}**: Good match! Consider emphasizing achievements or keywords from the JD.")

        # --- Optional download ---
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results CSV",
            csv,
            "job_match_results.csv",
            "text/csv"
        )
