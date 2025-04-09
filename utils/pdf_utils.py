
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
import base64
import io
import requests

def is_scanned_pdf(pdf_path: str) -> bool:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                return False
    return True

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def convert_pdf_to_image_base64(pdf_path: str) -> list[str]:
    images = convert_from_path(pdf_path, dpi=300)
    base64_images = []
    for image in images:
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)
    return base64_images

def send_to_ollama_node(text: str) -> dict:
    payload = {
        "model": "gemma:4b",
        "prompt": "Extraia os seguintes dados do contracheque: nome completo, matrícula, mês/ano de referência, e uma lista com código, descrição, percentual/duração e valor de cada vantagem. Formate a resposta em JSON.",
        "input": text,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    return response.json()
