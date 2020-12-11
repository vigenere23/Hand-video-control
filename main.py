import vlc_media_player as vmp


def main():
    vmp.setup()

    img = capture_image()
    segmented_img = segmentation(img)
    pred = classify(segmented_img)
    vmp.run(pred)


if __name__ == "__main__":
    main()
