FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api/main.py /app/main.py
COPY models/mushroom_catboost.cbm /app/models/mushroom_catboost.cbm

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]