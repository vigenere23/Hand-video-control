import cv2
import torch
from tqdm import tqdm
import numpy as np


def creating_images_save(data):
    number_pictures = data.shape[0]
    for i in tqdm(range(number_pictures)):
        img = np.zeros((28,28))
        compte = 1
        for line in range(28):
            for column in range(28):
                img[line, column] = int(data["pixel"+str(compte)][i])
                compte += 1
            cv2.imwrite("data/sign_mnist_test/test" + str(i) + ".png", img)
    return


def creating_images_array(data):
    train_img = []
    number_pictures = data.shape[0]
    for i in tqdm(range(number_pictures)):
        img = np.zeros((28,28))
        compte = 1
        for line in range(28):
            for column in range(28):
                img[line, column] = int(data["pixel"+str(compte)][i])
                compte += 1
        img /= 255.0
        img = img.astype("float32")
        train_img.append(img)
    return train_img


def images_to_tensor(images):
    images = np.array(images).reshape(int(images.shape[0]), 1, 28, 28)
    images: torch.Tensor = torch.from_numpy(images)
    return images


def normalize_image_input(image: np.ndarray):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 1, 28, 28).astype("float32") / 255
    return torch.from_numpy(image)


def get_square_boudaries(binary_image: np.ndarray, margins_percentage: float = 0.) -> tuple:
    points = cv2.findNonZero(binary_image)

    height = binary_image.shape[0]
    width = binary_image.shape[1]

    x, y, w, h = cv2.boundingRect(points)
    cx, cy = x + w/2, y + h/2

    if w < h:
        w = h
    w += width * margins_percentage

    x1 = max(0, int(cx - w/2))
    x2 = min(width, int(cx + w/2))
    y1 = max(0, int(cy - w/2))
    y2 = min(height, int(cy + w/2))

    return x1, x2, y1, y2
