"""
main_resnet_svr_linear.py
============================
FastAPI app that serves predictions from the resnet50_svr_linear bundle.

BUNDLE_DIR is read from an environment variable so the same image/code
works locally and in Docker/ECS without editing source. In the Dockerfile
this is set to /app/final_model/resnet50_svr_linear.
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from inference import MoisturePredictor, NotPotatoImageError

# ----------------------------------------------------------------------
# Config — override with the BUNDLE_DIR env var in Docker/ECS.
# Falls back to the local relative path for running on your machine.
# ----------------------------------------------------------------------
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "final_model/resnet50_svr_linear")

app = FastAPI(title="Potato Moisture Content Prediction API (ResNet50 + SVR-linear)")

predictor: MoisturePredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = MoisturePredictor(BUNDLE_DIR)
    print(f"Loaded model bundle from: {BUNDLE_DIR}")


class PredictionResponse(BaseModel):
    moisture_content: float
    model_used: str = "resnet50_svr_linear"


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()

    try:
        mc = predictor.predict_from_bytes(image_bytes)
    except NotPotatoImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(moisture_content=mc)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "bundle_dir": BUNDLE_DIR}