"""
AI Resume Screening & Job Recommendation — Streamlit Web App
Run with: streamlit run app.py & ngrok (see notebook Phase 11)
"""
import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# PAGE CONFIG & THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Job Recommendation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Force dark text everywhere - the original bug was backgrounds were
       pinned to light colors but text color was left to inherit Streamlit's
       theme. In dark mode that default text is near-white, giving invisible
       white-on-light text. Every rule below is !important so it always wins. */
    html, body { color: #1a1a1a !important; }
    .main { background-color: #f5f8fc; }
    .stApp { background-color: #f5f8fc; }
    [data-testid="stAppViewContainer"] { background-color: #f5f8fc; }
    [data-testid="stAppViewContainer"] * { color: #1a1a1a; }
    h1, h2, h3, h4, h5, h6 { color: #0b3d91 !important; }
    p, span, label, li, [data-testid="stMarkdownContainer"] * {
        color: #1a1a1a !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #0b3d91 !important;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(11, 61, 145, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #e3ecf9;
        color: #1a1a1a !important;
    }
    .card * { color: #1a1a1a !important; }
    .metric-card {
        background: linear-gradient(135deg, #0b3d91, #1e6fd9);
        color: white !important;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card * { color: white !important; }
    .job-table th { background-color: #0b3d91 !important; color: white !important; }
    .job-table td { color: #1a1a1a !important; }
    /* Sidebar keeps dark-blue bg + white text - must come after the generic
       rules above so its !important white wins for sidebar elements only. */
    section[data-testid="stSidebar"] { background-color: #0b3d91; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODEL_DIR = "Resume_AI_Project/models"
LABEL_MAPPING_PATH = "Resume_AI_Project/label_mapping.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert():
    from transformers import BertTokenizerFast, BertForSequenceClassification
    path = os.path.join(MODEL_DIR, "bert")
    tok = BertTokenizerFast.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    return tok, model

@st.cache_resource(show_spinner="Loading RoBERTa model...")
def load_roberta():
    from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
    path = os.path.join(MODEL_DIR, "roberta")
    tok = RobertaTokenizerFast.from_pretrained(path)
    model = RobertaForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    return tok, model

@st.cache_resource(show_spinner="Loading Sentence-BERT model...")
def load_sbert():
    from sentence_transformers import SentenceTransformer
    path = os.path.join(MODEL_DIR, "sentence_bert")
    return SentenceTransformer(path if os.path.exists(path) else "all-MiniLM-L6-v2")

@st.cache_data
def load_label_mapping():
    return pd.read_csv(LABEL_MAPPING_PATH)

# ----------------------------------------------------------------------------
# JOB CATALOG (mirrors Phase 7 in the notebook)
# ----------------------------------------------------------------------------
JOB_CATALOG = {
    "Data Scientist": "Analyze large datasets, build machine learning models, statistics, python, sql, data visualization, deep learning, predictive modeling.",
    "Data Analyst": "Interpret data, generate reports, dashboards, sql, excel, tableau, power bi, statistical analysis, business insights.",
    "Machine Learning Engineer": "Design and deploy machine learning models, python, tensorflow, pytorch, mlops, model deployment, feature engineering.",
    "AI Engineer": "Build artificial intelligence systems, deep learning, nlp, computer vision, neural networks, model optimization.",
    "Business Analyst": "Gather business requirements, process improvement, stakeholder management, sql, excel, reporting, documentation.",
    "Software Engineer": "Design and develop software applications, java, python, c++, data structures, algorithms, system design.",
    "Web Developer": "Build websites and web applications, html, css, javascript, react, node.js, rest api, responsive design.",
    "Full Stack Developer": "Frontend and backend development, react, node.js, mongodb, express, sql, api development, deployment.",
    "DevOps Engineer": "CI/CD pipelines, docker, kubernetes, aws, azure, infrastructure as code, automation, monitoring.",
    "Cloud Engineer": "Design cloud infrastructure, aws, azure, gcp, terraform, networking, security, scalability.",
    "Database Administrator": "Manage databases, sql, mysql, postgresql, oracle, backup recovery, performance tuning, data integrity.",
    "Network Engineer": "Design and maintain networks, routing, switching, firewalls, tcp/ip, network security, troubleshooting.",
    "Cybersecurity Analyst": "Protect systems from threats, penetration testing, vulnerability assessment, siem, incident response, security audits.",
    "UI/UX Designer": "Design user interfaces and experiences, figma, adobe xd, wireframes, prototypes, usability testing.",
    "Project Manager": "Plan and manage projects, agile, scrum, stakeholder communication, risk management, budgeting, timelines.",
    "HR Manager": "Recruitment, employee relations, performance management, payroll, policy development, onboarding.",
    "Digital Marketing Specialist": "SEO, SEM, social media marketing, content strategy, google analytics, email marketing, campaign management.",
    "Financial Analyst": "Financial modeling, budgeting, forecasting, excel, valuation, investment analysis, reporting.",
    "Mechanical Engineer": "Product design, CAD, autocad, solidworks, manufacturing processes, thermodynamics, prototyping.",
    "Civil Engineer": "Structural design, construction management, autocad, project planning, site supervision, building codes.",
}
JOB_TITLES = list(JOB_CATALOG.keys())
JOB_DESCRIPTIONS = list(JOB_CATALOG.values())

SKILL_KEYWORDS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "nosql", "mongodb",
    "react", "angular", "node.js", "django", "flask", "machine learning", "deep learning",
    "nlp", "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
    "numpy", "aws", "azure", "gcp", "docker", "kubernetes", "git", "html", "css",
    "tableau", "power bi", "excel", "hadoop", "spark", "linux", "rest api", "microservices",
    "data analysis", "data visualization", "statistics", "figma", "photoshop", "seo",
]
EDUCATION_KEYWORDS = [
    "b.tech", "btech", "bachelor", "m.tech", "mtech", "master", "phd", "doctorate",
    "b.sc", "bsc", "m.sc", "msc", "mba", "bca", "mca", "diploma", "b.e", "m.e",
]

# ----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------
def clean_resume_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif name.endswith(".docx"):
        import docx
        document = docx.Document(uploaded_file)
        return "\n".join([p.text for p in document.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def extract_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if 2 <= len(line.split()) <= 4 and not any(ch.isdigit() for ch in line):
            return line.title()
    return "Candidate"

def extract_skills(text):
    text_lower = text.lower()
    return sorted(set([s for s in SKILL_KEYWORDS if s in text_lower]))

def extract_education(text):
    text_lower = text.lower()
    return sorted(set([e for e in EDUCATION_KEYWORDS if e in text_lower]))

def extract_experience_years(text):
    patterns = [r'(\d+)\+?\s*years?\s*(of)?\s*experience', r'experience\s*(of)?\s*(\d+)\+?\s*years?']
    years_found = []
    text_lower = text.lower()
    for pattern in patterns:
        for m in re.findall(pattern, text_lower):
            for g in (m if isinstance(m, tuple) else (m,)):
                if g.isdigit():
                    years_found.append(int(g))
    return max(years_found) if years_found else 0

def predict_with_model(text, model, tokenizer, label_df, max_len=256):
    cleaned = clean_resume_text(text)
    encoding = tokenizer(cleaned, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
    category_row = label_df[label_df["label_id"] == pred_idx.cpu().item()]
    category = category_row["category"].values[0] if len(category_row) else "Unknown"
    return category, round(confidence.cpu().item() * 100, 2)

def recommend_jobs(resume_text, sbert_model, top_k=5):
    cleaned = clean_resume_text(resume_text)
    job_embeddings = sbert_model.encode(JOB_DESCRIPTIONS, convert_to_numpy=True)
    resume_embedding = sbert_model.encode([cleaned], convert_to_numpy=True)
    sims = cosine_similarity(resume_embedding, job_embeddings)[0]
    ranked = np.argsort(sims)[::-1][:top_k]
    return [{"job_title": JOB_TITLES[i], "similarity_score": round(float(sims[i]) * 100, 2)} for i in ranked]

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("🤖 Resume AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Model Comparison", "ℹ️ About Project"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Upload New Resume**")
sidebar_reset = st.sidebar.button("🔄 Reset / Upload New Resume")
if sidebar_reset:
    st.session_state.clear()
    st.rerun()

# ----------------------------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("AI Resume Screening & Job Recommendation")
    st.caption("Upload a resume to get instant AI-powered screening, classification, and job matching.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Resume", type=["pdf", "docx", "txt"])
    analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and analyze_clicked:
        with st.spinner("Extracting text and running AI models..."):
            raw_text = extract_text_from_file(uploaded_file)
            st.session_state["raw_text"] = raw_text
            st.session_state["name"] = extract_name(raw_text)

    if "raw_text" in st.session_state:
        raw_text = st.session_state["raw_text"]

        # ---------------- Resume Analysis Page ----------------
        st.header("📄 Resume Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Candidate Profile")
            st.write(f"**Name:** {st.session_state.get('name', 'Candidate')}")
            skills = extract_skills(raw_text)
            education = extract_education(raw_text)
            exp_years = extract_experience_years(raw_text)
            st.write(f"**Skills Detected:** {', '.join(skills) if skills else 'None found'}")
            st.write(f"**Education Detected:** {', '.join(education) if education else 'None found'}")
            st.write(f"**Experience Detected:** {exp_years} years")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Resume Summary")
            st.write(raw_text[:600] + ("..." if len(raw_text) > 600 else ""))
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- AI Prediction Section ----------------
        st.header("🧠 AI Prediction")
        try:
            label_df = load_label_mapping()
            bert_tok, bert_mdl = load_bert()
            roberta_tok, roberta_mdl = load_roberta()

            bert_cat, bert_conf = predict_with_model(raw_text, bert_mdl, bert_tok, label_df)
            roberta_cat, roberta_conf = predict_with_model(raw_text, roberta_mdl, roberta_tok, label_df)

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("BERT Prediction")
                st.write(f"**Category:** {bert_cat}")
                st.progress(int(bert_conf))
                st.write(f"Confidence: **{bert_conf}%**")
                st.markdown('</div>', unsafe_allow_html=True)
            with pcol2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("RoBERTa Prediction")
                st.write(f"**Category:** {roberta_cat}")
                st.progress(int(roberta_conf))
                st.write(f"Confidence: **{roberta_conf}%**")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Classification models not available yet: {e}")
            bert_cat, bert_conf, roberta_cat, roberta_conf = "N/A", 0, "N/A", 0

        # ---------------- Job Recommendation Section ----------------
        st.header("💼 Job Recommendations")
        sbert_mdl = load_sbert()
        recommendations = recommend_jobs(raw_text, sbert_mdl, top_k=5)
        rec_df = pd.DataFrame(recommendations).rename(
            columns={"job_title": "Recommended Job", "similarity_score": "Similarity Score (%)"}
        )
        st.table(rec_df)

        # ---------------- Match Score Card ----------------
        st.header("🎯 Overall Match Score")
        overall_match = round((bert_conf + roberta_conf + recommendations[0]["similarity_score"]) / 3, 2)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.progress(int(overall_match))
        st.markdown(
            f'<div class="metric-card"><h2>{overall_match}%</h2><p>Overall Resume-Job Match</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- Download Report Section ----------------
        st.header("📥 Download Report")
        report_text = f"""AI RESUME SCREENING REPORT
================================
Candidate: {st.session_state.get('name', 'Candidate')}

Skills Detected: {', '.join(skills) if skills else 'None'}
Education Detected: {', '.join(education) if education else 'None'}
Experience Detected: {exp_years} years

BERT Prediction: {bert_cat} ({bert_conf}% confidence)
RoBERTa Prediction: {roberta_cat} ({roberta_conf}% confidence)

Top Job Recommendations:
{rec_df.to_string(index=False)}

Overall Match Score: {overall_match}%
"""
        st.download_button(
            "⬇️ Download AI Recommendation Report",
            data=report_text,
            file_name="AI_Recommendation_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("👆 Upload a resume (PDF/DOCX/TXT) and click **Analyze Resume** to get started.")

# ----------------------------------------------------------------------------
# MODEL COMPARISON PAGE
# ----------------------------------------------------------------------------
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    comp_path = "Resume_AI_Project/models/model_comparison.csv"
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.table(comp_df)
    else:
        st.info("Run the notebook training phases first to generate model comparison data.")
    st.markdown("""
    **BERT (bert-base-uncased):** Bidirectional transformer, strong general-purpose contextual understanding.

    **RoBERTa (roberta-base):** Robustly optimized BERT variant, trained longer on more data, often yields
    higher accuracy on classification tasks.

    **Sentence-BERT (all-MiniLM-L6-v2):** Lightweight sentence embedding model used purely for semantic
    similarity-based job recommendation (not classification).
    """)

# ----------------------------------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------------------------------
else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### AI Agent for Resume Screening and Job Recommendation using Resume Parsing

    This project is an end-to-end AI recruitment assistant that:
    - Parses resumes (PDF/DOCX/TXT) and extracts skills, education, and experience.
    - Classifies resumes into job categories using fine-tuned **BERT** and **RoBERTa** models.
    - Recommends the top-5 best-fit job roles using **Sentence-BERT** semantic similarity.
    - Produces a downloadable AI recommendation report.

    **Tech Stack:** Python, PyTorch, HuggingFace Transformers, Sentence-Transformers, Streamlit, ngrok.

    **Dataset:** Kaggle Resume Dataset (gauravduttakiit/resume-dataset).

    Built for internship / academic project demonstration.
    """)
