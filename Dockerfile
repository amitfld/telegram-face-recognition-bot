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

# set the CMake policy flag globally for all subsequent RUN steps
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# upgrade pip
RUN pip install --upgrade pip

# now this pip install will pick up $CMAKE_ARGS in the build environment
RUN pip install dlib==19.24.2

# Now install everything else (WITHOUT dlib in requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt
# Copy bot code
COPY . .

# Run your bot
CMD ["python", "bot.py"]
