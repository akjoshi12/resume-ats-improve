# preprocessing.py
import re
from pdfminer.high_level import extract_text
from docx import Document

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'[^A-Za-z0-9#+%.,-]', ' ', text)  # Keep essential special chars
    return text.strip()

def preprocess_file(file_path):
    """Preprocess a PDF or DOCX file."""
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        print("Unsupported file format.")
        return ""
    return clean_text(text)

def preprocess_text(text):
    """Preprocess raw text input by cleaning it."""
    return clean_text(text)

# Example usage for testing preprocessing on a text string.
if __name__ == "__main__":
    sample_text = "   This is   a  sample text!!\n New line.\tTab included."
    preprocessed_text = preprocess_text(sample_text)
    print(preprocessed_text)
