# QFTT Bootstrap Loop Docker Container
# Based on Python 3.10 slim for efficiency

FROM python:3.10-slim

# Install system dependencies for QuTiP and numerical libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p data/sample_output data/benchmarks figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Default command runs a basic simulation
CMD ["python", "scripts/run_simulation.py", "--max_events", "100"]

# For Jupyter notebooks (uncomment if needed)
# EXPOSE 8888
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]