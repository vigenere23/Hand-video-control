import cv2
import torch
import time
import numpy as np
from cnn import load_model
from image import normalize_image_input, crop_square_region
from testing import predict_sign, find_sign_group
from segmentation_main import segmentation_contour, getContours
from vlc_media_player import VLCController

# Délai en secondes
DELAY = 1

def delay():
    "Time between A and B"
    b = time.time()
    if (b - delay.a) > delay.length:
        delay.a = b
        delay.process = True

delay.length = DELAY
delay.a = time.time()

def main():
    # Contrôle de VLC
    vlc_control = VLCController()

    # Use Webcam
    webcam = cv2.VideoCapture(0)  # Seule caméra est celle de l'ordi
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # id pour le nombre de pixel I guess
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # id pour le nombre de pixel I guess
    webcam.set(cv2.CAP_PROP_BRIGHTNESS, 75)  # id pour le brightness
    # Load CNN model
    model, *_ = load_model("0.954_Net_1607723644.4435968")

    while True:
        # Capture image from webcam
        sucess, image = webcam.read()
        capture = image.copy()

        # Return image of just the hand
        image_hand = segmentation_contour(image)

        # Modify images for the CNN model
        cv2.imshow("Image main", image_hand)
        sign = predict_sign(model, image_hand)
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

        # Send command to video
        delay()
        if delay.process:
            vlc_control.run(group)
        delay.process = False

        # Delay capture
        time.sleep(DELAY)


if __name__ == "__main__":
    main()
