import numpy as np
from cnn import load_model
from dataset import TestSignLanguageDataset
from testing import predict_sign
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def show_confusion_matrix(cm, target_names, title, save_path=None):
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

  tick_marks = np.arange(len(target_names))
  plt.xticks(tick_marks, target_names)
  plt.yticks(tick_marks, target_names)

  thresh = cm.max() / 2

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(
      j, i, "{:,}".format(cm[i, j]),
      horizontalalignment="center",
      color="white" if cm[i, j] > thresh else "black"
    )

  plt.tight_layout()

  plt.title(title)
  plt.ylabel('Vraie valeur')
  plt.xlabel('Prédiction')

  try:
    plt.show()
  except Exception as e:
    print(e)

  if save_path:
    fig.savefig(save_path)


def build_confusion_matrix(model, dataset):
  model.eval()

  predictions = model(dataset.images).argmax(dim=1)
  target = dataset.target

  return confusion_matrix(target, predictions)


def run():
  model = load_model('0.9392_HandyNet_1607802541.3999255')[0]
  test_dataset = TestSignLanguageDataset()
  
  predictions = model(test_dataset.images).argmax(dim=1)
  target = test_dataset.target

  cm = build_confusion_matrix(model, test_dataset)

  target_names = [chr(65+i) for i in range(25)]
  target_names.pop(9) # remove J

  show_confusion_matrix(cm, target_names, 'Matrice de confusion du réseau HandyNet', save_path="confusion_matrix.png")


if __name__ == "__main__":
  run()
