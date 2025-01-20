import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from textblob import TextBlob
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# OCR Setup (Ensure Tesseract is installed on your system)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from images
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to analyze text and provide suggestions
def analyze_text(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    sentiment = TextBlob(text).sentiment
    suggestions = []
    if sentiment.polarity < 0.5:
        suggestions.append("Consider using more positive language.")
    if len(keywords) < 5:
        suggestions.append("Add more specific and relevant keywords.")
    return keywords, sentiment, suggestions

# Streamlit App
st.title("Social Media Content Analyzer")

uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    st.subheader("Extracted Text")
    st.text(text)

    st.subheader("Analysis and Suggestions")
    if text:
        keywords, sentiment, suggestions = analyze_text(text)
        st.write("**Keywords:**", ", ".join(keywords))
        st.write("**Sentiment:**", f"Polarity: {sentiment.polarity:.2f}, Subjectivity: {sentiment.subjectivity:.2f}")
        st.write("**Suggestions:**")
        for suggestion in suggestions:
            st.write("- ", suggestion)
