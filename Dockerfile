FROM python:3.10-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt && \
    rm -rf /tmp/pip-* || true


# Command to run
CMD ["python", "app.py", "--real_time"]
