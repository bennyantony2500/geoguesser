from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import asyncio
import io

from model import (
    predict_streetclip,
    extract_text_from_image,
    search_country_for_text,
    fuse_predictions
)

app = FastAPI(title="GeoGuesser Ensemble API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "GeoGuesser Ensemble API v2 - POST an image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

    if file.content_type not in allowed:
        raise HTTPException(400, f"Invalid file type '{file.content_type}'. Use JPEG/PNG/WEBP")

    contents = await file.read()

    if not contents:
        raise HTTPException(400, "Empty file received")

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 10MB")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # FIX 1: was .converts()
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    loop = asyncio.get_event_loop()
    clip_task = loop.run_in_executor(None, predict_streetclip, image, 5)
    ocr_task  = loop.run_in_executor(None, extract_text_from_image, image)

    clip_preds, ocr_texts = await asyncio.gather(clip_task, ocr_task)

    ocr_result = await search_country_for_text(ocr_texts) if ocr_texts else None

    result = fuse_predictions(clip_preds, ocr_result, ocr_texts)

    return JSONResponse(content={          # FIX 2: was cotent
        **result,
        "streetclip_predictions": clip_preds,
    })