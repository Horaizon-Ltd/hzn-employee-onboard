FROM public.ecr.aws/lambda/python:3.11

RUN yum update -y && \
    yum install -y \
    wget \
    tar \
    gzip \
    make \
    autoconf \
    automake \
    libtool \
    pkgconfig \
    libpng-devel \
    libjpeg-devel \
    libtiff-devel \
    zlib-devel \
    poppler-utils \
    python3-devel \
    gcc10 \
    gcc10-c++ \
    && yum clean all

# Use GCC 10 for compilation (required for C++17 filesystem support in Tesseract 5.5.1)
ENV CC=/usr/bin/gcc10-cc
ENV CXX=/usr/bin/gcc10-c++

# Build Leptonica (Tesseract dependency)
WORKDIR /tmp
RUN wget http://www.leptonica.org/source/leptonica-1.82.0.tar.gz && \
    tar -xzf leptonica-1.82.0.tar.gz && \
    cd leptonica-1.82.0 && \
    ./configure --prefix=/opt && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf leptonica-1.82.0*

# Build Tesseract OCR 5.5.1
RUN wget https://github.com/tesseract-ocr/tesseract/archive/5.5.1.tar.gz && \
    tar -xzf 5.5.1.tar.gz && \
    cd tesseract-5.5.1 && \
    ./autogen.sh && \
    PKG_CONFIG_PATH=/opt/lib/pkgconfig ./configure --prefix=/opt && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf tesseract-5.5.1* 5.5.1.tar.gz

# Download Tesseract language data (English and Danish)
# Using tessdata_best for highest accuracy with special characters (æ, ø, å)
RUN mkdir -p /opt/share/tessdata && \
    cd /opt/share/tessdata && \
    wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata && \
    wget https://github.com/tesseract-ocr/tessdata_best/raw/main/dan.traineddata

# Verify Tesseract installation and version
RUN /opt/bin/tesseract --version

# Set environment for Tesseract
ENV LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/bin:$PATH
ENV TESSDATA_PREFIX=/opt/share/tessdata
ENV OMP_THREAD_LIMIT=1

# Copy requirements file
WORKDIR ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY src/ ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "main.lambda_handler" ]
