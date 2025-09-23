FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install python packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501

# Run the application
CMD ["python", "src/main.py"]