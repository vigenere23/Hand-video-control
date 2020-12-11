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


def crop_square_region(image: np.ndarray, points: np.ndarray, padding_percentage: float = 0.) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]

    x, y, w, h = cv2.boundingRect(points)
    cx, cy = x + w/2, y + h/2

    size = max(w, h) # so that a square fits all the boundaries

    if size > width or size > height:
        raise ValueError("a square region could not be cropped due to original image size")

    padding = width * padding_percentage
    size += padding

    padding += min(cx - size/2, 0)
    padding += min(width - cx - size/2, 0)
    padding += min(cy - size/2, 0)
    padding += min(height - cy - size/2, 0)

    size += padding

    x1 = max(0, int(cx - size/2))
    x2 = min(width, int(cx + size/2))
    y1 = max(0, int(cy - size/2))
    y2 = min(height, int(cy + size/2))

    image = image[y1:y2, x1:x2]

    return image
