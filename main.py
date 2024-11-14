import cv2
import torch
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from RealESRGAN import RealESRGAN
from PIL import Image
from subprocess import run
from deoldify.visualize import get_image_colorizer
import os
import time
from rembg import remove
import uuid

TOKEN = '7782074635:AAEzESEk-XVikiao30GI5V6RJSWmjVMs2EY'
BOT_USERNAME = '@aiiiieditorbot'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

colorizer = get_image_colorizer(artistic=True)
user_state = {}


def load_image_enhancer(scale=4):
    enhancer = RealESRGAN(device, scale=scale)
    enhancer.load_weights(f'weights/RealESRGAN_x{scale}.pth')
    return enhancer


def apply_image_enhancement(image, enhancer):
    return np.array(enhancer.predict(image)) if isinstance(enhancer.predict(image), Image.Image) else enhancer.predict(
        image)


def convert_image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def colorize_image(image_path):
    output_image = colorizer.get_transformed_image(image_path, render_factor=35)
    output_path = 'colorized_image.jpg'
    output_image.save(output_path)
    return output_path


def remove_image_background_with_options(image_path, background_type='transparent'):
    input_image = Image.open(image_path).convert("RGBA")
    removed_background = remove(input_image)

    if background_type == 'transparent':
        output_path = f'background_removed_transparent_{uuid.uuid4()}.png'
        removed_background.save(output_path, format="PNG")
        return output_path
    elif background_type in ['white', 'black']:
        background_color = (255, 255, 255, 255) if background_type == 'white' else (0, 0, 0, 255)
        bg_array = Image.new("RGBA", input_image.size, background_color)
        bg_array.paste(removed_background, mask=removed_background.split()[3])
        output_path = f'background_removed_{background_type}_{uuid.uuid4()}.png'
        bg_array.save(output_path, format="PNG")
        return output_path


def auto_adjust_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    accumulator = [float(hist[0])]
    for i in range(1, len(hist)):
        accumulator.append(accumulator[i - 1] + float(hist[i]))
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    minimum_gray = next(i for i in range(len(hist)) if accumulator[i] > clip_hist_percent)
    maximum_gray = next(i for i in range(len(hist) - 1, 0, -1) if accumulator[i] < (maximum - clip_hist_percent))
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


async def greet_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Улучшить качество", callback_data='enhance')],
        [InlineKeyboardButton("Сделать изображение черно-белым", callback_data='grayscale')],
        [InlineKeyboardButton("Раскрасить изображение", callback_data='colorize')],
        [InlineKeyboardButton("Создать портрет", callback_data='portrait')],
        [InlineKeyboardButton("Удалить фон (белый)", callback_data='remove_bg_white')],
        [InlineKeyboardButton("Удалить фон (черный)", callback_data='remove_bg_black')],
        [InlineKeyboardButton("Удалить фон (прозрачный)", callback_data='remove_bg_transparent')],
        [InlineKeyboardButton("Автояркость", callback_data='brightness')]  # Добавлен новый пункт
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"Привет! Я {BOT_USERNAME}, ваш AI-редактор изображений. Пожалуйста, выберите действие:",
        reply_markup=reply_markup)


async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_state[query.message.chat_id] = query.data
    await query.edit_message_text(f"Вы выбрали: {query.data}. Пожалуйста, отправьте изображение.")


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    file = await update.message.photo[-1].get_file()

    input_dir = './test_data/test_portrait_images/your_portrait_im'
    os.makedirs(input_dir, exist_ok=True)
    image_path = os.path.join(input_dir, 'input.jpg')
    await file.download_to_drive(image_path)

    if not os.path.exists(image_path):
        await update.message.reply_text("Ошибка: изображение не найдено.")
        return

    image = cv2.imread(image_path)
    action = user_state.get(chat_id, None)

    if action == 'enhance':
        enhancer = load_image_enhancer(scale=4)
        enhanced_image = apply_image_enhancement(image, enhancer)
        output_path = f'enhanced_image_{uuid.uuid4()}.jpg'
        cv2.imwrite(output_path, enhanced_image)
        with open(output_path, 'rb') as f:
            await update.message.reply_photo(photo=f)

    elif action == 'grayscale':
        grayscale_image = convert_image_to_grayscale(image)
        grayscale_image_path = f'grayscale_image_{uuid.uuid4()}.jpg'
        cv2.imwrite(grayscale_image_path, grayscale_image)
        with open(grayscale_image_path, 'rb') as f:
            await update.message.reply_photo(photo=f)

    elif action == 'colorize':
        colorized_image_path = colorize_image(image_path)
        with open(colorized_image_path, 'rb') as f:
            await update.message.reply_photo(photo=f)


    elif action == 'brightness':
        adjusted_image = auto_adjust_brightness_contrast(image)
        brightness_path = f'adjusted_brightness_{uuid.uuid4()}.jpg'
        cv2.imwrite(brightness_path, adjusted_image)
        with open(brightness_path, 'rb') as f:
            await update.message.reply_photo(photo=f)

    elif action in ['remove_bg_white', 'remove_bg_black', 'remove_bg_transparent']:
        background_type = action.split('_')[-1]
        bg_removed_path = remove_image_background_with_options(image_path, background_type)
        with open(bg_removed_path, 'rb') as f:
            await update.message.reply_document(document=f, filename=os.path.basename(bg_removed_path))

    elif action == 'portrait':
        await update.message.reply_text("Создаю портрет...")
        try:
            result = run(['python', 'U-2-Net/u2net_portrait_demo.py', image_path], capture_output=True, text=True)
            print("U-2-Net output:", result.stdout)
            print("U-2-Net errors:", result.stderr)
        except Exception as e:
            await update.message.reply_text(f"Ошибка при выполнении скрипта: {e}")
            return
        output_filename = 'input_portrait.png'
        output_path = os.path.join('test_data/test_portrait_images/your_portrait_results', output_filename)
        time.sleep(1)
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                await update.message.reply_photo(photo=f)
        else:
            await update.message.reply_text("Ошибка: не удалось найти портретное изображение.")
    del user_state[chat_id]


def main() -> None:
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", greet_user))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))
    application.add_handler(CallbackQueryHandler(handle_button_click))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
