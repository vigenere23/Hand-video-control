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
            cv2.imwrite("archive/sign_mnist_test/test" + str(i) + ".png", img)
    return


def bgr_image_to_tensor(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 1, 28, 28).astype("float32") / 255
    return torch.from_numpy(image)


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