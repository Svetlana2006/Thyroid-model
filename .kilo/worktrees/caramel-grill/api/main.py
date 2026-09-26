"""FastAPI application — CORS, routes, error handling."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.inference import predict
from api.registry import build_model_list
from api.schemas import HealthResponse, ModelsResponse, PredictResponse

app = FastAPI(
    title="Thyroid Nodule Classifier API",
    description="REST API for thyroid nodule malignancy prediction",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check — confirms backend is reachable and reports device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(status="ok", device=device)


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """Returns available models by scanning outputs/checkpoints/."""
    models = build_model_list()
    return ModelsResponse(models=models)


@app.post("/api/predict", response_model=PredictResponse)
async def run_prediction(
    image: UploadFile = File(...),
    model_id: str = Form(...),
):
    """Run inference on an uploaded image with the specified model."""
    # Validate model_id
    available = build_model_list()
    valid_ids = {m.id for m in available}
    if model_id not in valid_ids:
        raise HTTPException(status_code=404, detail="Model not found")

    # Read and validate image
    try:
        image_data = await image.read()
        if len(image_data) == 0:
            raise ValueError("Empty file")
        # Wrap in BytesIO for PIL
        image_io = io.BytesIO(image_data)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read image file")

    # Run prediction
    try:
        result = predict(image_io, model_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Catch architecture mismatches, corrupt checkpoints, etc.
        error_msg = str(e)
        if "state_dict" in error_msg.lower() or "mismatch" in error_msg.lower():
            raise HTTPException(
                status_code=500,
                detail="Could not load checkpoint — architecture mismatch",
            )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error_msg}")
