import os

from dotenv import load_dotenv

load_dotenv()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# for each model in the dotenv MODELS, build the GCS path variable
MODELS = os.getenv("MODELS").split(",")
GCS_MODEL_PATHS = [f"{model}_raw_5.keras" for model in MODELS]
GOOGLE_APPLICATION_CREDENTIALS = 'credentials/credentials.json'

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_DATABASE"),
}
