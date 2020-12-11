import cv2
import torch
from torch import nn
from cnn import load_model
from image import normalize_image_input, crop_square_region
from matplotlib import pyplot as plt
import numpy as np


LETTER_GROUPS = {
    'fist': ['A', 'E', 'M', 'N', 'O', 'S', 'X'],
    'pointing_up': ['X', 'U', 'R', 'K', 'D', 'I', 'T'],
    'opened': ['B', 'C'],
    'pointing_side': ['G', 'H'],
    'Y': ['Y'],
    'F': ['F'],
    'L': ['L'],
    'pointing_down': ['P', 'Q'],
    'peace': ['V', 'W']
}


def is_close_to_letter(predicted: str, truth: str) -> bool:
    for group in LETTER_GROUPS.values():
        if (truth in group) and (predicted in group):
            return True

    return False


def predict_sign(model: nn.Module, image: np.ndarray) -> str:
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    image = normalize_image_input(image)

    if torch.cuda.is_available():
        image = image.cuda()

    predicted_index = model(image).argmax(dim=1).item()
    predicted_letter = chr(65 + predicted_index)

    return predicted_letter


def find_sign_group(sign: str) -> str:
    for group, letters in LETTER_GROUPS.items():
        if sign in letters:
            return group
    
    return None


def test_realworld_images(model: nn.Module):
    results = {
        'exact': set(),
        'close': set(),
        'wrong': {}
    }
    percentages = []
    predictions = []

    LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    for letter in LETTERS:
        image = cv2.imread(f"test_images/{letter}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY_INV)
        points = cv2.findNonZero(binary_image)
        image = crop_square_region(image, points, padding_percentage=0.05)

        predicted_letter = predict_sign(model, image)
        predictions.append(predicted_letter)

        if predicted_letter == letter:
            results['exact'].add(letter)
        elif is_close_to_letter(predicted_letter, letter):
            results['close'].add(letter)
        else:
            results['wrong'][letter] = predicted_letter

    print()

    for result_type, letters in results.items():
        if result_type == 'wrong':
            percentage = round(float(len(letters.keys())) / float(len(LETTERS)) * 100, 1)
            percentages.append(percentage)
            errors = ', '.join(map(lambda kv: kv[0]+'>'+kv[1], letters.items()))
            keys = ', '.join(letters.keys())
            print(f"{result_type} ({percentage}%):\t{errors if percentage <= 50 else keys}")
        else:
            percentage = round(float(len(letters)) / float(len(LETTERS)) * 100, 1)
            percentages.append(percentage)
            print(f"{result_type} ({percentage}%):\t{', '.join(letters)}")

    print()

    predictions.sort()
    print(f"predictions:\t{', '.join(predictions)}")
    print(f"unique pred.:\t{', '.join(set(predictions))}")

    return percentages


def run():
    model, *_ = load_model('0.8807_Net_1607714720.6292076')
    test_realworld_images(model)


if __name__ == "__main__":
    run()
