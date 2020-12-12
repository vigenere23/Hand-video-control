import time

import cv2
import numpy as np
import torch

from cnn import load_model
from image import crop_square_region, normalize_image_input
from segmentation_main import getContours, segmentation_contour
from testing import find_sign_group, predict_sign
from vlc_media_player import VLCController


def main():
    # Contrôle de VLC
    vlc_control = VLCController(DELAY=3)

    # Use Webcam
    webcam = cv2.VideoCapture(0)  # Seule caméra est celle de l'ordi
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # id pour le nombre de pixel I guess
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # id pour le nombre de pixel I guess
    webcam.set(cv2.CAP_PROP_BRIGHTNESS, 75)  # id pour le brightness
    # Load CNN model
    model, *_ = load_model("0.9351_Net_1607783575.9996943")

    while True:
        # Capture image from webcam
        sucess, image = webcam.read()
        capture = cv2.flip(image.copy(), flipCode=1)
        sign = None

        # Return image of just the hand
        try:
            image_hand = segmentation_contour(image)
            cv2.imshow("Image main", image_hand)
            print("Taille image: {}".format(image_hand.shape))
            sign = predict_sign(model, image_hand, threshold=0.9)
        except Exception as e:
            print(e)

        group = find_sign_group(sign)
        
        capture = cv2.putText(
            capture,
            f"{sign} {str(group)}",
            (10, capture.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            thickness=3,
        )
        cv2.imshow("Image", capture)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        vlc_control.run(group)


if __name__ == "__main__":
    main()
