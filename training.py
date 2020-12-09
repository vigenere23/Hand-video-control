import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cnn import Net, GoogleNet, save_model, load_model
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
    path = f"data/{dataset_type}.plt"

    try:
        images, target = torch.load(path)
    except Exception as e:
        print(e)
        print("Loading from csv... this might take a while")
        csv_path = f"archive/sign_mnist_{dataset_type}/sign_mnist_{dataset_type}.csv"
        images, target = load_csv_dataset(csv_path)
        torch.save([images, target], path)

    return images, target


class SignLanguageDataset():
    def __init__(self, dataset_type: str):
        self.__images, self.__target = load_dataset(dataset_type)
        self.transform = nn.Sequential(
            transforms.RandomResizedCrop((28, 28), scale=(0.9, 1.1), ratio=(9./10., 10./9.)),
            transforms.RandomRotation(10)
        )

    def __len__(self):
        return self.__target.shape[0]

    def __getitem__(self, index):
        return self.__images[index], self.__target[index]


def calculate_accuracy(output, target):
    output_classes = output.argmax(dim=1)
    accuracy = ((output_classes == target).float().sum() / target.shape[0])
    return accuracy


def train_epoch(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, train_gen: DataLoader):
    losses = []
    accuracies = []
    model.train()

    for batch, (images, target) in enumerate(train_gen):
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
            images = images.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        # print("Batch: ", batch, "\t", "loss: ", loss.item(), "\t", "accuracy: ", accuracy.item())

    epoch_loss = np.array(losses).mean()
    epoch_accuracy = np.array(accuracies).mean()

    print("TRAIN", "\t", "loss: ", epoch_loss, "\t", "accuracy: ", epoch_accuracy)

    return epoch_loss, epoch_accuracy


def test_epoch(model: nn.Module, criterion: nn.Module, test_gen: DataLoader):
    losses = []
    accuracies = []
    model.eval()

    with torch.no_grad():
        for images, target in test_gen:
            if torch.cuda.is_available():
                model = model.cuda()
                criterion = criterion.cuda()
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


def train(model: nn.Module, n_epochs: int, batch_size: int, optimizer: optim.Optimizer, criterion: nn.Module, train_dataset: Dataset, test_dataset: Dataset):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    model_to_save = {}

    train_gen = DataLoader(train_dataset, batch_size, shuffle=True)
    test_gen = DataLoader(test_dataset, batch_size, shuffle=False)

    for epoch_number in range(n_epochs):
        train_loss_sum = 0
        print(f"\n------ EPOCH {epoch_number} ------\n")

        train_loss, train_accuracy = train_epoch(model, optimizer, criterion, train_gen)
        test_loss, test_accuracy = test_epoch(model, criterion, test_gen)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

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
    n_epochs = 30
    batch_size = 128
    lr = 0.0003

    # Loading dataset
    train_dataset = SignLanguageDataset("train")
    test_dataset = SignLanguageDataset("test")

    # TODO create Dataloader to load in batches

    # Creating model
    model = Net(26)
    # model = GoogleNet(train_y.max() + 1)

    # Creating learning modules
    model_parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model_parameters, lr = lr)
    criterion = nn.CrossEntropyLoss()

    train(model, n_epochs, batch_size, optimizer, criterion, train_dataset, test_dataset)



if __name__ == "__main__":
    run()
