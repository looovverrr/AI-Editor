# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch (если необходимо для твоего проекта)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Клонируем репозитории DeOldify и Real-ESRGAN
RUN git clone https://github.com/jantic/DeOldify.git DeOldify
RUN git clone https://github.com/xinntao/Real-ESRGAN.git Real-ESRGAN

# Устанавливаем зависимости для DeOldify
WORKDIR /DeOldify
RUN pip install -r requirements-colab.txt  # Используй этот файл, если работаешь в Colab

# Устанавливаем зависимости для Real-ESRGAN
WORKDIR /Real-ESRGAN
RUN pip install -r requirements.txt

# Копируем requirements.txt и устанавливаем зависимости для остальных библиотек
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота в контейнер
COPY . /main
WORKDIR /main

# Устанавливаем веса для моделей DeOldify и Real-ESRGAN
# Можно загрузить их вручную или скриптом, например:
# RUN mkdir /main/models
# RUN wget <URL для весов> -O /main/models/model_weights.pth

# Запускаем бота
CMD ["python", "bot.py"]
