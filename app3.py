import streamlit as st
import pdfplumber
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

skill_keywords = ["php", "html", "css", "javascript", "excel", "communication", "training", "machine learning"]
exp_keywords = ["internship", "project", "experience", "years"]
edu_keywords = ["b.tech", "m.tech", "college", "university", "school"]

def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text if text.strip() else "No text found in PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def extract_info(text):
    info = {"skills": [], "experience": "", "education": ""}
    for line in text.split(". "):
        if any(skill in line for skill in skill_keywords):
            info["skills"].extend([skill for skill in skill_keywords if skill in line])
        if info["experience"] == "" and any(word in line for word in exp_keywords):
            info["experience"] = line
        if info["education"] == "" and any(word in line for word in edu_keywords):
            info["education"] = line
    info["skills"] = list(set(info["skills"]))
    return info

def list_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0
    set1 = set([i.lower() for i in list1])
    set2 = set([i.lower() for i in list2])
    return round(len(set1.intersection(set2)) / max(len(set2), 1), 2)

def text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    return round(len(set1.intersection(set2)) / max(len(set2), 1), 2)

def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    cosine_sim = (tfidf * tfidf.T).toarray()[0, 1]
    return round(cosine_sim, 2)

def sbert_similarity(text1, text2):
    embedding1 = sbert_model.encode(text1, convert_to_tensor=True)
    embedding2 = sbert_model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding1, embedding2)
    return round(score.item(), 2)

def combine_scores(*scores):
    return round(np.mean(scores) * 100, 2)

def match_decision(score):
    return "Match" if score >= 50 else "Not a Match"

# Streamlit UI
st.set_page_config(page_title="Resume-JD Matcher", layout="centered")
st.title("Resume and Job Description Matcher")

resume_file = st.file_uploader("Upload Resume", type="pdf")
jd_file = st.file_uploader("Upload Job Description", type="pdf")

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(jd_text)

    resume_info = extract_info(resume_clean)
    jd_info = extract_info(jd_clean)

    skill_score = list_similarity(resume_info["skills"], jd_info["skills"])
    edu_score = text_similarity(resume_info["education"], jd_info["education"])
    exp_score = text_similarity(resume_info["experience"], jd_info["experience"])

    tfidf_score = tfidf_similarity(resume_clean, jd_clean)
    bert_score = sbert_similarity(resume_clean, jd_clean)

    overall_score = combine_scores(skill_score, edu_score, exp_score, tfidf_score, bert_score)

    # Human-readable output
    st.markdown("### Similarity Scores")
    st.markdown(f" **Skills Match**: {skill_score * 100:.2f}%")
    st.markdown(f" **Education Match**: {edu_score * 100:.2f}%")
    st.markdown(f" **Experience Match**: {exp_score * 100:.2f}%")
    st.markdown(f" **TF-IDF Similarity**: {tfidf_score * 100:.2f}%")
    st.markdown(f" **Sentence-BERT Similarity**: {bert_score * 100:.2f}%")
    st.markdown(f" **Custom Overall Score**: {overall_score:.2f} / 100")
    st.markdown(f"### Match Decision: **{match_decision(overall_score)}**")

    with st.expander("Structured Resume Info"):
        st.json(resume_info)

    with st.expander("Structured JD Info"):
        st.json(jd_info)

    with st.expander("Raw Resume Text"):
        st.text_area("Resume", resume_text, height=200)

    with st.expander("Raw JD Text"):
        st.text_area("Job Description", jd_text, height=200)

else:
    st.info("Please upload both Resume and JD PDFs to continue.")
