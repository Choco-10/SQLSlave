FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Python 3.10, pip, git (for HuggingFace model download)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install PyTorch with the exact cu126 index matching requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.12.0+cu126 \
    torchvision==0.27.0+cu126 \
    torchaudio==2.11.0+cu126

# Copy and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application source code
COPY . /app
WORKDIR /app

# Expose API port (FastAPI) and UI port (Streamlit)
EXPOSE 8000 8501

# Default command: serve FastAPI backend
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]