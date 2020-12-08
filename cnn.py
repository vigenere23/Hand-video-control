import time
import torch
from torch import nn, optim
from torchvision.models import googlenet


class Net(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 7*7, int(nb_classes))
        )
    
    def forward(self,x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


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


def load_model(path: str):
  data = torch.load(path)

  model = data['model']
  model.load_state_dict(data['model_state'])
  optimizer = data['optimizer']
  optimizer.load_state_dict(data['optimizer_state'])
  criterion = data['criterion']
  criterion.load_state_dict(data['criterion_state'])
  test_accuracy = data['test_accuracy']

  return model, optimizer, criterion, test_accuracy


def save_model(model: nn.Module = None, optimizer: optim.Optimizer = None, criterion: nn.Module = None, test_accuracy: float = None):
  model_name = model.__class__.__name__
  timestamp = time.time()
  test_accuracy = round(test_accuracy, 4)
  path = f"models/{test_accuracy}_{model_name}_{timestamp}.plt"

  data = {
    'model': model,
    'model_state': model.state_dict(),
    'optimizer': optimizer,
    'optimizer_state': optimizer.state_dict(),
    'criterion': criterion,
    'criterion_state': criterion.state_dict(),
    'test_accuracy': test_accuracy
  }

  torch.save(data, path)

  return path
