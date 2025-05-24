FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Create directory for data
RUN mkdir -p /app/data

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA support and PyTorch Geometric
RUN pip install --no-cache-dir torch==2.7.0 && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html && \
    pip install --no-cache-dir torch_geometric

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Make startup script executable
RUN chmod +x /app/scripts/startup.sh

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=0

# Command to run when container starts
CMD ["/app/scripts/startup.sh"] 