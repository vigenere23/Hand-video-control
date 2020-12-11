import cv2
import torch
import time
from cnn import load_model
from image import normalize_image_input, crop_square_region
from testing import predict_sign


def run():
    model, *_ = load_model('0.9713_Net_1607701742.3824296')
    cap = cv2.VideoCapture(0)

    while True:
        _, capture = cap.read()
        capture = cv2.flip(capture, 1)

        sign = predict_sign(model, capture)
        
        capture = cv2.putText(capture, sign, (10, capture.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
        cv2.imshow('live feed', capture)
        cv2.waitKey(1)


if __name__ == "__main__":
    run()
