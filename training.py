import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from cnn import Net, GoogleNet, save_model, load_model
from image import images_to_tensor, creating_images_array
from analyzing import test_realworld_images
import cv2
import os
from barbar import Bar


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
    except Exception as e:
        print(e)
        print("Loading from csv... this might take a while")
        csv_path = os.path.join("data", f"sign_mnist_{dataset_type}.csv")
        images, target = load_csv_dataset(csv_path)
        torch.save([images, target], path)

    return images, target


class SignLanguageDataset():
    def __init__(self, dataset_type: str):
        self.__images, self.__target = load_dataset(dataset_type)
        # self.__images = np.repeat(self.__images[..., np.newaxis], 3, -1)
        # self.__images = np.transpose(self.__images, (0, 1, 4, 2, 3))

        self.transform = nn.Sequential(
            # TODO normalize?
            transforms.RandomResizedCrop((28, 28), scale=(0.8, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(10, scale=(0.8, 1), fillcolor=1)
            # transforms.Resize(224)
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


def calculate_accuracy(output, target):
    output_classes = output.argmax(dim=1)
    accuracy = ((output_classes == target).float().sum() / target.shape[0])
    return accuracy


def split_dataset(dataset: Dataset, factor: float = 0.5):
    first_dataset_length = int(len(dataset) * factor)
    second_dataset_length = len(dataset) - first_dataset_length
    return random_split(dataset, [first_dataset_length, second_dataset_length])


def train_epoch(model: nn.Module, train_gen: DataLoader, val_gen: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: object = None, verbose: bool = False):
    losses = []
    accuracies = []
    model.train()

    train_gen = Bar(train_gen) if verbose else train_gen

    for batch, (images, target) in enumerate(train_gen):
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # print("Batch: ", batch, "\t", "loss: ", loss.item())

    val_gen = Bar(val_gen) if verbose else val_gen
    with torch.no_grad():

        for images, target in val_gen:
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
            
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)

            losses.append(loss.item())
            accuracies.append(accuracy.item())


    epoch_loss = np.array(losses).mean()
    epoch_accuracy = np.array(accuracies).mean()

    if scheduler is not None:
        scheduler.step(epoch_loss)

    print("\nVAL", "\t", "loss: ", epoch_loss, "\t", "accuracy: ", epoch_accuracy)

    return epoch_loss, epoch_accuracy


def test_epoch(model: nn.Module, test_gen: DataLoader, criterion: nn.Module):
    losses = []
    accuracies = []
    model.eval()

    with torch.no_grad():
        for images, target in test_gen:
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
                
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)

            losses.append(loss.item())
            accuracies.append(accuracy.item())

    epoch_loss = np.array(losses).mean()
    epoch_accuracy = np.array(accuracies).mean()

    print("TEST",  "\t", "loss: ", epoch_loss, "\t", "accuracy: ", epoch_accuracy)

    return epoch_loss, epoch_accuracy


def train(model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, val_dataset: Dataset, n_epochs: int, batch_size: int, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: object = None, verbose: bool = False):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    model_to_save = {}

    train_gen = DataLoader(train_dataset, batch_size, shuffle=True)
    val_gen = DataLoader(val_dataset, batch_size, shuffle=False)
    test_gen = DataLoader(test_dataset, batch_size, shuffle=False)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_number in range(n_epochs):
        train_loss_sum = 0
        print(f"\n------ EPOCH {epoch_number+1} ------\n")

        train_loss, train_accuracy = train_epoch(model, train_gen, val_gen, optimizer, criterion, scheduler=scheduler, verbose=verbose)
        test_loss, test_accuracy = test_epoch(model, test_gen, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if verbose:
            test_realworld_images(model)

        if (model_to_save.get('test_accuracy', 0) < test_accuracy):
            model_to_save = {
                'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'test_accuracy': test_accuracy
            }

    print("\n------ SUMMARY ------\n")
    print(f"Max train accuracy: {np.array(train_accuracies).max()}")
    print(f"Max test accuracy: {np.array(test_accuracies).max()}")

    save_path = save_model(**model_to_save)
    print(f"\nSaved best iteration to {save_path}")


def run():
    # Parameters
    n_epochs = 50
    batch_size = 128
    lr = 0.005

    # Loading dataset
    train_dataset = SignLanguageDataset("train")
    train_dataset, val_dataset = split_dataset(train_dataset, factor=0.8)
    test_dataset = SignLanguageDataset("test")

    # TODO create Dataloader to load in batches

    # Creating model
    model = Net(25)
    # model = GoogleNet(25)

    # Creating learning modules
    model_parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model_parameters, lr = lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=0.00001, verbose=True)

    train(model, train_dataset, val_dataset, test_dataset, n_epochs, batch_size, optimizer, criterion, scheduler=scheduler, verbose=True)



if __name__ == "__main__":
    run()
