import cv2
import torch
from torch import nn
from cnn import load_model
from image import bgr_image_to_tensor
from matplotlib import pyplot as plt
import numpy as np


def test_realworld_images(model: nn.Module):
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    letters = ['A', 'L', 'Q']

    for letter in letters:
        img = cv2.imread(f"test_images/{letter}.jpg")
        img = bgr_image_to_tensor(img)

        img = img.cuda()

        sign_index = model(img).argmax(dim=1).item()
        sign = chr(65 + sign_index)

        if sign == letter:
            print(f"I'm genius, I know that a {sign} is a {letter}")
        else:
            print(f"I'm dumb, I think that a {letter} is a {sign}")


def run():
    model, *_ = load_model('1.0_Net_1607547806.270809')
    test_realworld_images(model)


if __name__ == "__main__":
    run()
