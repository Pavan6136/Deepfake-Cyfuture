from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
from tensorflow import keras
from typing import Dict

app = FastAPI()


model = None
last_prediction = None  #helps us to Store the last prediction

@app.on_event("startup")

def load_model():
    """ helps in Loading model"""
    global model
    model = keras.models.load_model("/home/pavan/cyfuture/deepfake_model.keras")

@app.get("/")
def read_root():
    return {"message": "Deepfake Image Detection API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    """Predicts if an image is real or fake and stores the result"""
    global last_prediction  

    # to read and process the imge
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # to resize
    img = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0) 

    # Predict
    prediction = model.predict(img, verbose=0)[0][0]
    last_prediction = "Real" if prediction > 0.5 else "Fake"

    return {"prediction": last_prediction}

@app.get("/last_prediction/")
def get_last_prediction() -> Dict[str, str]:
    """Retrieves the last stored prediction"""
    if last_prediction is None:
        return {"message": "No predictions made yet"}
    return {"last_prediction": last_prediction}

#uncomment it if you want to directly run as python file .
#if __name__ == "__main__":
 #   uvicorn.run(app, host="0.0.0.0", port=8080)


