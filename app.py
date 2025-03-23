# app.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
import numpy as np
import io
from inference import evaluate
from PIL import Image


# Initialize FastAPI app and basic authentication
app = FastAPI()
security = HTTPBasic()


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "admin"
    correct_password = "password"  # Ideally, hash this password
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

def predict_image(image: UploadFile):
    image_data = image.file.read()
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')
    img = img.resize((32, 32))  # Resize while keeping the aspect ratio
    predicted_class = evaluate(img)
    return predicted_class

@app.post("/predict")
async def predict(credentials: HTTPBasicCredentials = Depends(security), image: UploadFile = File(...)):
    authenticate(credentials)
    predicted_class = predict_image(image)
    return {"predicted_class": predicted_class}
