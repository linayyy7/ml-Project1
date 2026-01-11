# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Download dataset (if not included)
# Uncomment if you want to download automatically
# RUN curl -L -o data/airline_passengers_satisfaction.csv https://example.com/dataset.csv

# Train the model (this will also create the model file)
RUN python train.py

# Expose port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9696/health || exit 1

# Run the application
CMD ["python", "predict.py"]