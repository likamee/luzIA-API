import os
import tempfile

from config import GCS_BUCKET_NAME, GCS_MODEL_PATHS, MODELS
from db import log_result
from fastapi import APIRouter, File, UploadFile
from models import evaluate_models, get_cam, get_img_array, load_model_from_gcs

router = APIRouter(prefix="", tags=["Base Routes"])

models_names = MODELS

models = [load_model_from_gcs(GCS_BUCKET_NAME, model_path) for model_path in GCS_MODEL_PATHS]

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if image_bytes is None:
        return {"error": "Invalid image"}


    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(image_bytes)
        temp.flush()

        # Use the get_img_array function to process the image
        image = get_img_array(temp.name, (299, 299))

        # Remove the temporary file
        os.unlink(temp.name)

    predictions = evaluate_models(models, image)

    # predictions is a list of numpy arrays, so we need to check if in each array we have any value
    # greater than 0.8. If we do, we have a concise prediction, so we will generate the cam for the
    # models that reach this threshold.

    # mount the response object as JSON as first level with the models (cataract, dr, etc), second level
    # with the predictions (0.8, 0.2, etc) and third level with the CAMs (image)
    response = {}
    # Generate CAM for the models in the list that surpasses the threshold, assuming it's a VGG16 model
    for idx, model in enumerate(models_names):
        response[model] = {}  # Initialize the nested dictionary for the current model
        response[model]['prediction'] = predictions[idx][0].tolist()
        if predictions[idx][0][1] > 0.8:
            response[model]['cam'] = get_cam(models[idx], image)
        else:
            response[model]['cam'] = None

    return response
