import os
import tempfile

from config import GCS_BUCKET_NAME, GCS_MODEL_PATHS, MODELS
from db import log_result
from fastapi import APIRouter, File, UploadFile, Form
from models import evaluate_model, get_cam, get_img_array, load_model_from_gcs

router = APIRouter(prefix="", tags=["Base Routes"])

models_names = MODELS

models = {name: load_model_from_gcs(GCS_BUCKET_NAME, model_path) for name, model_path in zip(models_names, GCS_MODEL_PATHS)}

@router.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    image_bytes = await file.read()

    if image_bytes is None:
        return {"error": "Invalid image"}


    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(image_bytes)
        temp.flush()

        # Use the get_img_array function to process the image
        image = get_img_array(temp.name, (299, 299))

    if model not in models:
        return {"error": "Invalid model"}

    prediction = evaluate_model(models[model], image)

    # mount the response object as JSON as first level with the models (cataract, dr, etc), second level
    # with the predictions (0.8, 0.2, etc) and third level with the CAMs (image)
    response = {}

    response[model] = {}  # Initialize the nested dictionary for the current model
    response[model]['prediction'] = prediction[0].tolist()
    if prediction[0][1] > 0.8:
        response[model]['cam'] = get_cam(models[model], image, temp.name)  # Use models[model] instead of models[idx]
    else:
        response[model]['cam'] = None

    os.unlink(temp.name)

    return response
