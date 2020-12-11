import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from image import images_to_tensor, creating_images_array


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


class TrainSignLanguageDataset():
    def __init__(self):
        self.__images, self.__target = load_dataset("train")

        self.transform = nn.Sequential(
            # TODO normalize?
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10, translate=(0.2, 0.2), scale=(0.8, 1.2), fillcolor=255),
            # transforms.GaussianBlur(3)
        )

    def __len__(self):
        return self.__target.shape[0]

    def __getitem__(self, index):
        image = self.__images[index]
        image = transforms.ToPILImage()(image)
        image = self.transform(image)
        image = transforms.ToTensor()(image)

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
