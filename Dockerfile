# Use lightweight Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face expects app on port 7860
ENV PORT=7860

CMD ["python", "app.py"]
