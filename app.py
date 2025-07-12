import streamlit as st
import pdfplumber

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

st.set_page_config(page_title="TalentAlign AI - Resume Matcher", layout="centered")

st.title("📄 TalentAlign AI")
st.subheader("Streamlining Talent Discovery with Intelligent Resume Matching")

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume")
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd")

if resume_file and jd_file:
    st.success("Files uploaded successfully!")

    with st.expander("📄 View Resume Text"):
        resume_text = extract_text_from_pdf(resume_file)
        st.text_area("Resume Content", resume_text, height=300)

    with st.expander("📝 View Job Description Text"):
        jd_text = extract_text_from_pdf(jd_file)
        st.text_area("JD Content", jd_text, height=300)
