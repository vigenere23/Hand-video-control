import cv2
import torch
import time
import numpy as np
from cnn import load_model
from image import normalize_image_input, crop_square_region
from testing import predict_sign, find_sign_group


def run():
    model, *_ = load_model('0.8807_Net_1607714720.6292076')
    cap = cv2.VideoCapture(0)

    while True:
        _, capture = cap.read()
        capture = cv2.flip(capture, 1)
        border = 100
        size = capture.shape[0]
        points = np.array([
            [border, border],
            [border, size - border],
            [size - border, size - border],
            [size - border, border],
        ])
        x, y, w, h = cv2.boundingRect(points)

        image = crop_square_region(capture, points)
        sign = predict_sign(model, image)
        group = find_sign_group(sign)
        
        capture = cv2.rectangle(capture, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        capture = cv2.putText(capture, f"{sign} {str(group)}", (10, capture.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
        cv2.imshow('live feed', capture)
        cv2.waitKey(1)


if __name__ == "__main__":
    run()
