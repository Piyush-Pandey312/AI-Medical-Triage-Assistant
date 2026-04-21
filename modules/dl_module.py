# Day 2 — DL Module
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io

MODEL_PATH = "/content/drive/MyDrive/fdp/dl_model.keras"
CLASS_MAP  = {0: "NORMAL", 1: "PNEUMONIA"}

print("Loading DL model from Drive...")
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
print("✅ DL model loaded")

def classify_xray(img_bytes):
    """
    Input  : img_bytes — raw image bytes (jpg/png)
    Output : { "label": "PNEUMONIA", "confidence": 0.91 }
    """
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr  = np.array(img, dtype=np.float32)
    arr  = preprocess_input(arr)
    arr  = np.expand_dims(arr, axis=0)

    pred      = model.predict(arr, verbose=0)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return {
        "label"     : CLASS_MAP[class_idx],
        "confidence": round(confidence, 2)
    }
