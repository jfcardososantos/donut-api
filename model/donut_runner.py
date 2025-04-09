
import base64
from utils.pdf_utils import is_scanned_pdf, convert_pdf_to_image_base64, extract_text_from_pdf, send_to_ollama_node
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
import io

# Carregamento do modelo Donut
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa").half()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def run_donut(image_base64: str) -> dict:
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((768, 768))

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    task_prompt = "<s_docvqa><s_question>Extraia os seguintes dados do contracheque: nome completo, matrícula, mês/ano de referência, e uma lista com código, descrição, percentual/duração e valor de cada vantagem.<s_answer>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=1536)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    try:
        json_data = json.loads(result[result.find("{"):])
    except:
        json_data = {"raw": result, "error": "Failed to parse JSON"}
    return json_data

def extract_data_from_pdf(pdf_path: str) -> dict:
    if is_scanned_pdf(pdf_path):
        images_base64 = convert_pdf_to_image_base64(pdf_path)
        return run_donut(images_base64[0])
    else:
        text = extract_text_from_pdf(pdf_path)
        return send_to_ollama_node(text)
