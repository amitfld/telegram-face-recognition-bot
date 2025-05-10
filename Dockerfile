FROM python:3.11-slim

# Install system packages required for dlib and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# First install the prebuilt dlib wheel
RUN pip install --upgrade pip && \
    pip install "https://github.com/RPi-Distro/python-dlib/releases/download/19.24.2/dlib-19.24.2-cp311-cp311-manylinux_2_28_x86_64.whl"




# Now install other Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy bot code
COPY . .

# Run your bot
CMD ["python", "bot.py"]
