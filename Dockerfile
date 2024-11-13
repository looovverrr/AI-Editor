# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные библиотеки
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота в контейнер
COPY . /main
WORKDIR /main

# Запускаем бота
CMD ["python", "bot.py"]
