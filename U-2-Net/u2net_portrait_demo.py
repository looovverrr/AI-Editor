import cv2
import torch
from model import U2NET
from torch.autograd import Variable
import numpy as np
from glob import glob
import os
import torch.nn.functional as F

def clear_output_directory(out_dir):
    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def detect_single_face(face_cascade, img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            print("Warning: No face detected, processing the whole image.")
            return None
        wh = 0
        idx = 0
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            if wh < w * h:
                idx = i
                wh = w * h
        return faces[idx]
    except Exception as e:
        print("Error in detect_single_face:", e)
        return None

def crop_face(img, face):
    if face is None:
        return img
    try:
        (x, y, w, h) = face
        height, width = img.shape[0:2]
        left, top = max(x - int(w * 0.4), 0), max(y - int(h * 0.6), 0)
        right, bottom = min(x + w + int(w * 0.4), width), min(y + h + int(h * 0.2), height)
        im_face = img[top:bottom, left:right]
        im_face = cv2.resize(im_face, (512, 512), interpolation=cv2.INTER_AREA)
        return im_face
    except Exception as e:
        print("Error in crop_face:", e)
        return None

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def inference(net, input):
    try:
        tmpImg = np.zeros((input.shape[0], input.shape[1], 3))
        input = input / np.max(input)
        tmpImg[:, :, 0] = (input[:, :, 2] - 0.406) / 0.225
        tmpImg[:, :, 1] = (input[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (input[:, :, 0] - 0.485) / 0.229
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = tmpImg[np.newaxis, :, :, :]
        tmpImg = torch.from_numpy(tmpImg).type(torch.FloatTensor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmpImg = tmpImg.to(device)
        d1, *_ = net(tmpImg)
        pred = 1.0 - d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = pred.squeeze().cpu().data.numpy()
        return pred
    except Exception as e:
        print("Error in inference:", e)
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_list = glob('D:/diplom/test_data/test_portrait_images/your_portrait_im/*')
    print("Number of images found:", len(im_list))

    out_dir = 'D:/diplom/test_data/test_portrait_images/your_portrait_results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        clear_output_directory(out_dir)

    face_cascade_path = 'U-2-Net/saved_models/face_detection_cv2/haarcascade_frontalface_default.xml'
    model_dir = 'U-2-Net/saved_models/u2net_portrait/u2net_portrait.pth'

    # Проверка путей
    if not os.path.isfile(face_cascade_path):
        print(f"Error: Haarcascade file not found at {face_cascade_path}")
        return
    if not os.path.isfile(model_dir):
        print(f"Error: Model file not found at {model_dir}")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    net = U2NET(3, 1)

    try:
        net.load_state_dict(torch.load(model_dir, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    net.to(device)
    net.eval()

    for i, img_path in enumerate(im_list):
        print(f"\nProcessing image {i + 1}/{len(im_list)}: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print("Warning: Unable to read image. Skipping...")
            continue

        print(f"Image shape: {img.shape}")  # Check the shape of the loaded image
        face = detect_single_face(face_cascade, img)
        im_face = crop_face(img, face)
        if im_face is None:
            print("Error: Failed to crop face.")
            continue

        im_portrait = inference(net, im_face)
        if im_portrait is None:
            print("Error: Failed to create portrait image.")
            continue

        output_path = os.path.join(out_dir, f"{os.path.basename(img_path)[:-4]}_portrait.png")
        if im_portrait is not None and im_portrait.size > 0:
            im_portrait = (im_portrait * 255).astype(np.uint8)
            cv2.imwrite(output_path, im_portrait)
            if os.path.exists(output_path):
                print(f"Portrait successfully saved at: {output_path}")
            else:
                print("Error: Portrait file was not saved as expected.")
        else:
            print("Error: Portrait image is None or empty, cannot save.")

if __name__ == '__main__':
    main()
