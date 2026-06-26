FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn pandas numpy pydantic
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
