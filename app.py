import streamlit as st
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

abbreviation_dict = {
    "js": "javascript",
    "html": "hypertext markup language",
    "css": "cascading style sheets",
    "sql": "structured query language",
    "dbms": "database management system",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "oops": "object oriented programming system",
    "api": "application programming interface"
}

def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text if text.strip() else " No text found in PDF."
    except Exception as e:
        return f" Error reading PDF: {e}"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    replaced_words = [abbreviation_dict.get(word, word) for word in filtered_words]
    return ' '.join(replaced_words)

st.set_page_config(page_title="TalentAlign AI - Resume Matcher", layout="centered")

st.title("TalentAlign AI")
st.subheader(" Intelligent Resume Matching")

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume")
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd")

if resume_file and jd_file:
    st.success("Files uploaded successfully")

    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    with st.expander("View Resume Text"):
        st.text_area("Resume Content", resume_text, height=300)

    with st.expander("View Job Description Text"):
        st.text_area("Job Description Content", jd_text, height=300)

    processed_resume_text = preprocess_text(resume_text)
    processed_jd_text = preprocess_text(jd_text)

    with st.expander("Preprocessed Resume Text"):
        st.text_area("Cleaned Resume Content", processed_resume_text, height=300)

    with st.expander("Preprocessed Job Description Text"):
        st.text_area("Cleaned Job Description Content", processed_jd_text, height=300)
