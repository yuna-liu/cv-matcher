# pages/2_📄_Job_Matcher.py

import io
import json
import time
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

# ---- OpenAI client helper ----
# Prefer secrets; fall back to a password input so the page works standalone.
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        st.error("OpenAI package not found. Add `openai` to requirements.txt.")
        st.stop()

    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    if not api_key:
        api_key = st.text_input("🔐 OpenAI API Key", type="password", help="Stored locally for this session only.")
        if not api_key:
            st.info("Add your API key (or configure in `.streamlit/secrets.toml`).", icon="🗝️")
            st.stop()

    return OpenAI(api_key=api_key)

# ---- Simple document readers ----
ALLOW_DOCX = False  # set True if you install `python-docx` in requirements

def read_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception as e:
        return ""

def read_docx(file) -> str:
    if not ALLOW_DOCX:
        return ""
    try:
        import docx  # python-docx
    except Exception:
        return ""
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception:
        return ""

def extract_cv_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(file)
    if name.endswith(".docx") and ALLOW_DOCX:
        return read_docx(file)
    return ""

# ---- LLM prompt & call ----
SYSTEM_INSTRUCTIONS = """You are an expert recruiter and technical screener.
You compare a candidate CV to a Job Description and return STRICT JSON ONLY.
Score fairly: consider required skills, recency, depth, domain fit, and responsibilities.
The score must be an integer 0–100.
"""

USER_TEMPLATE = """Compare the following candidate to the job.

JOB DESCRIPTION:
{jd}

CANDIDATE CV TEXT:
{cv}

Return ONLY a compact JSON object with the following exact keys:
{{
  "score": <integer 0-100>,
  "top_strengths": [<up to 6 short bullets>],
  "gaps": [<up to 6 short bullets>],
  "summary": "<2-3 concise sentences on fit>"
}}"""

def evaluate_cv(client, jd_text: str, cv_text: str) -> Dict[str, Any]:
    """Call OpenAI to get a structured JSON evaluation."""
    if not cv_text.strip():
        return {
            "score": 0,
            "top_strengths": [],
            "gaps": ["Could not extract any text from the CV (PDF parsing issue)."],
            "summary": "No readable content extracted."
        }

    prompt = USER_TEMPLATE.format(jd=jd_text.strip(), cv=cv_text[:15000])  # safety clip
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # good quality / cost tradeoff
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        content = resp.choices[0].message.content.strip()
        # Try strict JSON parse; if model adds prose, find JSON block.
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start:end+1])
            else:
                raise

        # Normalize + validate
        score = int(max(0, min(100, int(data.get("score", 0)))))
        strengths = [s.strip() for s in (data.get("top_strengths") or []) if str(s).strip()][:6]
        gaps = [g.strip() for g in (data.get("gaps") or []) if str(g).strip()][:6]
        summary = str(data.get("summary", "")).strip()

        return {
            "score": score,
            "top_strengths": strengths,
            "gaps": gaps,
            "summary": summary
        }
    except Exception as e:
        return {
            "score": 0,
            "top_strengths": [],
            "gaps": [f"Model error: {e}"],
            "summary": "An error occurred while evaluating this CV."
        }

# ---- UI ----
st.title("🎯 Job Matcher — Rank CVs Against a Job")
st.caption("Upload up to 5 CVs (PDF recommended), paste a job description, and get ranked matches + a downloadable report.")

client = get_openai_client()

uploaded_cvs = st.file_uploader(
    "Upload up to 5 CVs (PDF only for best results)",
    type=["pdf"] + (["docx"] if ALLOW_DOCX else []),
    accept_multiple_files=True
)

job_description = st.text_area(
    "Paste the Job Description",
    placeholder="Paste the job title, responsibilities, required skills, and nice-to-haves…",
    height=200
)

col_a, col_b = st.columns([1, 1])
with col_a:
    run_btn = st.button("🔎 Analyze & Rank")
with col_b:
    sample_btn = st.button("Use a sample JD")

if sample_btn and not job_description.strip():
    st.session_state["jd_sample"] = """We are hiring a Data Engineer with strong experience in Python, SQL, and Snowflake.
Responsibilities include building robust ETL/ELT pipelines (Matillion preferred), orchestrating workflows in Azure Data Factory,
and creating analytics-ready datasets for BI (Power BI). Experience with Databricks, PySpark, and CI/CD is a plus.
Knowledge of GenAI/RAG and LLM app integration is nice-to-have."""
    job_description = st.session_state["jd_sample"]
    st.experimental_rerun()

if run_btn:
    if not uploaded_cvs:
        st.error("Please upload at least one CV.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    results: List[Dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    for i, cv_file in enumerate(uploaded_cvs, start=1):
        status.write(f"Processing **{cv_file.name}** ({i}/{len(uploaded_cvs)})…")
        cv_text = extract_cv_text(cv_file)
        eval_data = evaluate_cv(client, job_description, cv_text)

        results.append({
            "CV File": cv_file.name,
            "Score": eval_data["score"],
            "Summary": eval_data["summary"],
            "Top Strengths": " • ".join(eval_data["top_strengths"]) if eval_data["top_strengths"] else "",
            "Gaps": " • ".join(eval_data["gaps"]) if eval_data["gaps"] else "",
        })

        progress.progress(i / len(uploaded_cvs))
        time.sleep(0.1)

    status.empty()
    progress.empty()

    if not results:
        st.warning("No results produced.")
        st.stop()

    # Rank and show
    df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # human-friendly rank

    st.subheader("🏆 Ranked Results")
    st.dataframe(
        df.style.format({"Score": "{:.0f}"}).background_gradient(
            subset=["Score"], cmap="Greens"
        ),
        use_container_width=True,
        hide_index=False
    )

    # Details per CV
    st.markdown("### 📂 Candidate Details")
    for idx, row in df.iterrows():
        with st.expander(f"{idx}. {row['CV File']} — {row['Score']}% match"):
            st.write(f"**Summary:** {row['Summary'] or '—'}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Strengths**")
                if row["Top Strengths"]:
                    for b in row["Top Strengths"].split(" • "):
                        st.markdown(f"- {b}")
                else:
                    st.write("—")
            with col2:
                st.markdown("**Gaps**")
                if row["Gaps"]:
                    for b in row["Gaps"].split(" • "):
                        st.markdown(f"- {b}")
                else:
                    st.write("—")

    # ---- Downloads ----
    st.markdown("### ⬇️ Download Report")
    csv_bytes = df.to_csv(index_label="Rank").encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="job_match_results.csv",
        mime="text/csv",
    )

    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index_label="Rank", sheet_name="Results")
    st.download_button(
        "Download Excel",
        data=xls_buf.getvalue(),
        file_name="job_match_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Footer tip
st.caption("Tip: PDF text quality matters. Export CVs as text-based PDFs (not scanned images).")
