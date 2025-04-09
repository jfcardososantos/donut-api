from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import io
import tempfile
import os

app = FastAPI()

# Carregando modelo Donut
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Função: Detecta se PDF tem texto extraível (não escaneado)
def is_text_based_pdf(file: UploadFile) -> bool:
    try:
        pdf_reader = PdfReader(file.file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
        return False
    except Exception:
        return False

# Função: Extrai imagens de um PDF escaneado (usando PyMuPDF)
def extract_images_from_pdf(file_bytes: bytes) -> List[Image.Image]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    zoom = 2.0  # Qualidade
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images

# Função: Processa imagem com Donut
def process_with_donut(image: Image.Image) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)
    task_prompt = "<s_docvqa><s_question>Extraia todo o conteúdo visível do documento com máxima fidelidade.</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=2048, early_stopping=True)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

@app.post("/infer")
async def infer(pdf: UploadFile = File(...)):
    if not pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é um PDF válido.")

    # Reposiciona ponteiro do arquivo
    contents = await pdf.read()
    pdf.file.seek(0)

    # Detecta se é um PDF digital com texto
    is_digital = is_text_based_pdf(pdf)

    # Se for digital, retorna o texto extraído direto
    if is_digital:
        pdf.file.seek(0)
        reader = PdfReader(pdf.file)
        text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
        return JSONResponse(content={"raw_text": text.strip()})

    # Caso contrário, OCR com Donut (escaneado)
    images = extract_images_from_pdf(contents)
    all_text = []
    for img in images:
        text = process_with_donut(img)
        all_text.append(text)

    joined_text = "\n\n".join(all_text)
    return JSONResponse(content={"raw_text": joined_text.strip()})
