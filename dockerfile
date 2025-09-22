# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app.py .
COPY model.py .
COPY model.pth .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
