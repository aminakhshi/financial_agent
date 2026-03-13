FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user for runtime
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /home/appuser/.local/share && chown -R appuser:appuser /home/appuser

# Install python packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Expose ports
EXPOSE 8501 8000

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV HOME=/home/appuser

USER appuser

# Run the application
CMD ["python", "src/main.py"]
