FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and training entrypoint
COPY src ./src
COPY train.py verify_pipeline.py ./

# Expose Gradio port
EXPOSE 7860

# Run Gradio app
CMD ["python", "-m", "src.ui.gradio_app"]
