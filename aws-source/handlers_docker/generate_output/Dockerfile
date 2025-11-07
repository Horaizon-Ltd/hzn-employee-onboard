# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Install build dependencies and runtime dependencies
RUN yum update -y && \
    yum install -y \
    wget \
    tar \
    gzip \
    gcc \
    gcc-c++ \
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
    && yum clean all

# Build Leptonica (Tesseract dependency)
WORKDIR /tmp
RUN wget http://www.leptonica.org/source/leptonica-1.82.0.tar.gz && \
    tar -xzf leptonica-1.82.0.tar.gz && \
    cd leptonica-1.82.0 && \
    ./configure --prefix=/opt && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf leptonica-1.82.0*

# Build Tesseract OCR
RUN wget https://github.com/tesseract-ocr/tesseract/archive/5.3.0.tar.gz && \
    tar -xzf 5.3.0.tar.gz && \
    cd tesseract-5.3.0 && \
    ./autogen.sh && \
    PKG_CONFIG_PATH=/opt/lib/pkgconfig ./configure --prefix=/opt && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf tesseract-5.3.0*

# Download Tesseract language data (English and Danish)
RUN mkdir -p /opt/share/tessdata && \
    cd /opt/share/tessdata && \
    wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata && \
    wget https://github.com/tesseract-ocr/tessdata/raw/main/dan.traineddata

# Set environment for Tesseract
ENV LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/bin:$PATH
ENV TESSDATA_PREFIX=/opt/share/tessdata

# Copy requirements file
WORKDIR ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY src/ ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "main.lambda_handler" ]
