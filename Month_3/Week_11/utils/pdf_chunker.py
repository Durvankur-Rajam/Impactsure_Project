import fitz
import re


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=512, overlap=64):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def process_pdf(pdf_path, chunk_size=512, overlap=64):
    text = extract_text(pdf_path)
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Extracted {len(chunks)} chunks from {pdf_path}")
    return chunks