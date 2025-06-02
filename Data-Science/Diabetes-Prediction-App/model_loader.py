import boto3
import joblib
import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS S3 Configuration from .env
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = 'diabetes-ml-model-bucket'
MODEL_FILE = 'best_ann_model.h5'
SCALER_FILE = 'scaler.pkl'

# Local download paths
LOCAL_MODEL_PATH = f"models/{MODEL_FILE}"
LOCAL_SCALER_PATH = f"models/{SCALER_FILE}"

def download_from_s3(file_name, local_path):
    """Download a file from AWS S3"""
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(BUCKET_NAME, file_name, local_path)

def load_model_and_scaler():
    """Download and load ANN model + scaler from S3"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_from_s3(MODEL_FILE, LOCAL_MODEL_PATH)
    if not os.path.exists(LOCAL_SCALER_PATH):
        download_from_s3(SCALER_FILE, LOCAL_SCALER_PATH)

    model = load_model(LOCAL_MODEL_PATH)
    scaler = joblib.load(LOCAL_SCALER_PATH)
    return model, scaler
