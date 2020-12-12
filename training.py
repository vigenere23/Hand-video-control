import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from cnn import Net, GoogleNet, save_model, load_model
from testing import test_realworld_images
from dataset import load_dataset, split_dataset, TrainSignLanguageDataset, TestSignLanguageDataset
from barbar import Bar


def calculate_accuracy(output, target):
    output_classes = output.argmax(dim=1)
    accuracy = ((output_classes == target).float().sum() / target.shape[0])
    return accuracy


def train_epoch(model: nn.Module, train_gen: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, verbose: bool = False):
    model.train()

    for images, target in Bar(train_gen):
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if verbose:
            print("\t", "loss: ", loss.item())


def validate_epoch(model: nn.Module, val_gen: DataLoader, criterion: nn.Module, scheduler: object = None):
    losses = []
    accuracies = []
    model.train()
    
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


def train(model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, val_dataset: Dataset, n_epochs: int, batch_size: int, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: object = None, real_world_test: bool = False):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_model = {}

    train_gen = DataLoader(train_dataset, batch_size, shuffle=True)
    val_gen = DataLoader(val_dataset, batch_size, shuffle=False)
    test_gen = DataLoader(test_dataset, batch_size, shuffle=False)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_number in range(n_epochs):
        train_loss_sum = 0
        print(f"\n------ EPOCH {epoch_number+1} ------\n")

        train_epoch(model, train_gen, optimizer, criterion, verbose=False)
        train_loss, train_accuracy = validate_epoch(model, val_gen, criterion, scheduler=scheduler)
        test_loss, test_accuracy = test_epoch(model, test_gen, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if real_world_test:
            test_realworld_images(model)

        if (best_model.get('test_accuracy', 0) < test_accuracy):
            best_model = {
                'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'test_accuracy': test_accuracy
            }

    last_model = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'test_accuracy': test_accuracy
    }

    print("\n------ SUMMARY ------\n")
    print(f"Max train accuracy: {np.array(train_accuracies).max()}")
    print(f"Max test accuracy: {np.array(test_accuracies).max()}")

    best_model_save_path = save_model(**best_model)
    # last_model_save_path = save_model(**last_model)
    print(f"\nSaved best iteration to {best_model_save_path}")
    # print(f"\nSaved last iteration to {last_model_save_path}")


def run():
    # Parameters
    n_epochs = 30
    batch_size = 256
    lr = 0.01

    # Loading dataset
    train_dataset = TrainSignLanguageDataset()
    train_dataset, val_dataset = split_dataset(train_dataset, factor=0.9)
    test_dataset = TestSignLanguageDataset()

    # Creating model
    model = Net(25)
    # model = GoogleNet(25)

    # Creating learning modules
    model_parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model_parameters, lr = lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=0.00001, verbose=True)

    train(model, train_dataset, val_dataset, test_dataset, n_epochs, batch_size, optimizer, criterion, scheduler=scheduler, real_world_test=True)



if __name__ == "__main__":
    run()
