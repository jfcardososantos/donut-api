
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model.donut_runner import extract_data_from_pdf
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/infer")
async def infer_from_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        json_result = extract_data_from_pdf(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(file_path)

    return json_result
