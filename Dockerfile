# Используем базовый образ Python 3.10
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем gdown для работы с Google Drive
RUN pip install --no-cache-dir gdown

# Устанавливаем PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Копируем файл requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем папку для весов и скачиваем их
RUN mkdir -p /app/weights

# Скачать веса из Google Drive
RUN gdown --id 1Gy09UPByypbKz9xfHU_iPvMc34yGEFwJ -O /app/weights/RealESRGAN_x4.pth \
    && gdown --id 1Z_D0kAytAP7QAaar5k9FLH7ZDE1cuVt4 -O /app/weights/other_weights_1.pth \
    && gdown --id 1qmXEyXyiJU07jhJm_3e-lT2hOYg0_QNx -O /app/weights/other_weights_2.pth \
    && gdown --id 107AHBtMafUZFCr-tZVTob-JrMHUi_On_ -O /app/weights/other_weights_3.pth \
    && gdown --id 1qtwplC96TwpYqKPl-wNxWnqeW0Tn-6kQ -O /app/weights/other_weights_4.pth

# Копируем код бота в контейнер
COPY . /main
WORKDIR /main

# Запускаем бота
CMD ["python", "bot.py"]
