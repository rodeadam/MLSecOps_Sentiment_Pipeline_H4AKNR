# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_docker.txt .

# Install Python dependencies (Updated: 2025-12-08 - MLflow 2.10.2)
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy application files
COPY app.py .
COPY MLModel.py .
COPY constants.py .
COPY streamlit_app.py .

# Create directories for MLflow and reports
RUN mkdir -p /app/mlruns /app/mlartifacts /app/reports

# Expose ports
EXPOSE 8080 5102 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "app.py"]