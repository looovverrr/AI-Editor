import os
import glob
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import io
from PIL import Image
import numpy as np
from model import U2NET  # Убедись, что здесь правильно импортируется модель U-2-Net
from data_loader import SalObjDataset, RescaleT, ToTensorLab


# Нормализуем предсказанную карту вероятностей
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_output(image_name, pred, d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    img_name = os.path.basename(image_name)
    img_name_without_ext = os.path.splitext(img_name)[0]

    imo.save(os.path.join(d_dir, f"{img_name_without_ext}_segmented.png"))


def inference_human_segmentation(image_path, model):
    """Запуск сегментации на изображении."""
    img_name_list = [image_path]  # Обернём в список для обработки

    # Создаем датасет и загрузчик данных для изображения
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )

    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Выполняем инференс
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _, _, _, _, _, _ = model(inputs_test)

        # Нормализация
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # Сохранение результатов
        prediction_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images_results')
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        save_output(image_path, pred, prediction_dir)


def load_model():
    """Загрузка модели U2-Net."""
    model_name = 'u2net_human_seg'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    return net


# Глобальная загрузка модели
model = load_model()


# Использование в функции обработки изображения бота
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    file = await update.message.photo[-1].get_file()
    image_path = 'input.jpg'
    await file.download_to_drive(image_path)

    # Выполнение удаления фона
    inference_human_segmentation(image_path, model)

    # Отправка результата обратно пользователю
    result_image_path = os.path.join(os.getcwd(), 'test_data', 'test_human_images_results',
                                     os.path.basename(image_path).split('.')[0] + '_segmented.png')
    with open(result_image_path, 'rb') as f:
        await update.message.reply_photo(photo=f)

    # Опционально, удалить входные и выходные изображения после обработки
    os.remove(image_path)
    if os.path.exists(result_image_path):
        os.remove(result_image_path)
