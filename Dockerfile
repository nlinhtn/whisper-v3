FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Cài đặt dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python cần thiết
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .

# Expose port
EXPOSE 8080

# Khởi động server
CMD ["python3", "app.py"]