# Base image
FROM python:3.12-slim

WORKDIR /app

# Copy only source code and requirements
COPY requirements.txt .
COPY app.py .
COPY model.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
