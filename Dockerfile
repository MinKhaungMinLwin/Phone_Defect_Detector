FROM python:3.10-slim

WORKDIR /app

# Install OpenCV dependency
RUN apt-get update && apt-get install -y libgl1

# Install system dependencies (optional but common)
RUN apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install python-multipart  # 👈 Add this line

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
