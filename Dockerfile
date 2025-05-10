FROM python:3.11-slim

# 1) Install system deps required by dlib/face_recognition
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

WORKDIR /app

# 2) Export CMAKE_ARGS so pip’s build backend sees it
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# 3) Upgrade pip and install dlib (will pick up $CMAKE_ARGS)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir dlib==19.24.2

# 4) Install remaining dependencies (make sure requirements.txt does not include dlib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy your bot’s code and set the default command
COPY . .
CMD ["python", "bot.py"]
