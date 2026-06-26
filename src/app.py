from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import os

# Определяем абсолютные пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

# Загружаем модель один раз при старте сервера
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris ML API", description="API для предсказания сорта Ириса")


# Редирект на документацию
@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


# Контракт данных
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Эндпоинт для проверки здоровья
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running"}


# Эндпоинт для предсказания
@app.post("/predict")
def predict(features: IrisFeatures):
    data = [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]
    ]
    prediction = model.predict(data)
    return {"predicted_class": int(prediction[0])}
