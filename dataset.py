import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from image import images_to_tensor, creating_images_array
import cv2


def load_csv_dataset(csv_path: str):
    dataset = pd.read_csv(csv_path)

    images = images_to_tensor(creating_images_array(dataset))

    target = dataset['label'].values.astype(int)
    target = torch.from_numpy(target)

    return images, target


def load_dataset(dataset_type: str):
    images = None
    target = None
    path = os.path.join("data", f"{dataset_type}.plt")

    try:
        images, target = torch.load(path)
    except IOError as e:
        print("Loading dataset from csv... this might take a while")
        csv_path = os.path.join("data", f"sign_mnist_{dataset_type}.csv")
        images, target = load_csv_dataset(csv_path)
        torch.save([images, target], path)

    return images, target


def split_dataset(dataset: Dataset, factor: float = 0.5):
    first_dataset_length = int(len(dataset) * factor)
    second_dataset_length = len(dataset) - first_dataset_length
    return random_split(dataset, [first_dataset_length, second_dataset_length])


class Transform(nn.Module):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x)


class TrainSignLanguageDataset():
    def __init__(self):
        self.__images, self.__target = load_dataset("train")

        self.transform = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize((0.5), (0.5)),
            # transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
            transforms.RandomErasing(scale=(0.1, 0.33), ratio=(2./5., 5./2.)),
            # transforms.RandomAffine(10, translate=(0.2, 0.2), scale=(0.7, 1.1), fillcolor=255),
            Transform(transforms.ToPILImage()),
            transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), fillcolor=255),
            Transform(transforms.ToTensor())
        )

    def __len__(self):
        return self.__target.shape[0]

    def __getitem__(self, index):
        image = self.__images[index]
        image = self.transform(image)

        # cv2.imshow('1', image.numpy()[0])
        # cv2.waitKey(400)

        target = self.__target[index]

        return image, target


class TestSignLanguageDataset():
    def __init__(self):
        self.__images, self.__target = load_dataset("test")

    def __len__(self):
        return self.__target.shape[0]

    def __getitem__(self, index):
        image = self.__images[index]
        target = self.__target[index]

        return image, target
