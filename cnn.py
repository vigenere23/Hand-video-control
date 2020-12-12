import time
import torch
from torch import nn, optim


class HandyNetLayer(nn.Module):
    def __init__(self, size: int, channel_in: int, channel_out: int, kernel_size: int = 3, padding: int = 0, max_pooling: bool = False):
        super().__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.size = size - (kernel_size - 1) + 2*padding

        modules = [
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
        ]

        if max_pooling:
            modules.append(nn.MaxPool2d(kernel_size=2))
            self.size = int(self.size / 2)

        self.model = nn.Sequential(*modules)

    def forward(self,x):
        return self.model(x)

    def add_next(self, channel_out: int, kernel_size: int = 3, padding: int = 0, max_pooling: bool = False):
        return HandyNetLayer(self.size, self.channel_out, channel_out, kernel_size=kernel_size, padding=padding, max_pooling=max_pooling)


class HandyNet(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()

        conv1 = HandyNetLayer(28, 1, 32)
        conv2 = conv1.add_next(64, max_pooling=True)
        conv3 = conv2.add_next(128, padding=1, max_pooling=True)
        conv4 = conv3.add_next(128, padding=1, max_pooling=True)
        full1 = nn.Linear(conv4.size ** 2 * conv4.channel_out, 256)
        full2 = nn.Linear(256, int(nb_classes))

        self.model = nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            nn.Flatten(),
            full1,
            nn.ReLU(inplace=True),
            full2
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
