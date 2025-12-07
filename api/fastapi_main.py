"""
FastAPI application exposing image processing operations:
predict, resize, grayscale, and normalize.
"""

import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image, UnidentifiedImageError

from mylib.model import (
    predict_class,
    resize_image,
    convert_to_grayscale,
    normalize_image,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Load the home HTML page."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict randomly the class of an image."""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        prediction = predict_class(img)
        return {"prediction": prediction}
    except (ValueError, UnidentifiedImageError):
        return {"error": "Invalid image"}


@app.post("/resize")
async def resize(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
):
    """Resize an image."""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        resized = resize_image(img, (width, height))
        return {"width": resized.size[0], "height": resized.size[1]}
    except (ValueError, UnidentifiedImageError):
        return {"error": "Invalid input"}


@app.post("/grayscale")
async def grayscale(file: UploadFile = File(...)):
    """Convert an image to grayscale."""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        gray = convert_to_grayscale(img)
        return {"mode": gray.mode}
    except (ValueError, UnidentifiedImageError):
        return {"error": "Invalid input"}


@app.post("/normalize")
async def normalize(file: UploadFile = File(...)):
    """Normalize image pixels."""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        data = normalize_image(img)
        return {
            "min": float(data.min()),
            "max": float(data.max()),
        }
    except (ValueError, UnidentifiedImageError):
        return {"error": "Invalid input"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.fastapi_main:app", host="127.0.0.1", port=8000, reload=True)
