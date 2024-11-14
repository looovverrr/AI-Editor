# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    wget \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch (CPU-версия)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Загрузка весов для RealESRGAN
RUN mkdir -p /app/weights && \
    wget -O /app/weights/RealESRGAN_x4.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x4plus.pth

# Копируем код приложения
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Запуск приложения
CMD ["python", "bot.py"]
