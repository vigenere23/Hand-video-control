import time
import torch
from torch import nn, optim
from torchvision.models import googlenet


class Net(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, 128, kernel_size = 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 25, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6),
            nn.Flatten()
        )

        self.model.apply(self.__init_weights)

    def __init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.)
        elif type(module) == nn.Conv2d:
            nn.init.kaiming_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self,x):
        return self.model(x)


class GoogleNet(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()

        self.model = googlenet(pretrained=True)
        dim_before_fc = self.model.fc.in_features

        for param in self.model.parameters():
                        param.requires_grad = False

        self.model.fc = nn.Linear(dim_before_fc, int(nb_classes))

    def forward(self, x):
        print(x.shape)
        return self.model.forward(x)


def load_model(filename: str):
    data = torch.load(f"models/{filename}.plt", map_location='cpu')

    model: nn.Module = data['model']
    model.load_state_dict(data['model_state'])
    optimizer: optim.Optimizer = data['optimizer']
    optimizer.load_state_dict(data['optimizer_state'])
    criterion: nn.Module = data['criterion']
    criterion.load_state_dict(data['criterion_state'])
    test_accuracy: float = data['test_accuracy']

    return model, optimizer, criterion, test_accuracy


def save_model(model: nn.Module = None, optimizer: optim.Optimizer = None, criterion: nn.Module = None, test_accuracy: float = None):
    model_name = model.__class__.__name__
    timestamp = time.time()
    test_accuracy = round(test_accuracy, 4)
    path = f"models/{test_accuracy}_{model_name}_{timestamp}.plt"

    data = {
        'model': model.cpu(),
        'model_state': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state': optimizer.state_dict(),
        'criterion': criterion.cpu(),
        'criterion_state': criterion.state_dict(),
        'test_accuracy': test_accuracy
    }

    torch.save(data, path)

    return path
