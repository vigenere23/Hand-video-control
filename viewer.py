import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

num_to_letter = lambda num: chr(ord("a") - 1 + num)

# CSV data
df_train = pd.read_csv(os.path.join("data", "sign_mnist_train.csv"))
df_test = pd.read_csv(os.path.join("data", "sign_mnist_test.csv"))

y_train = df_train["label"]
y_test = df_train["label"]

# Remove label
df_train = df_train.drop("label", axis=1)
df_test = df_test.drop("label", axis=1)

X_train = df_train.values
X_test = df_test.values

# Show figure
fig, ax = plt.subplots(3, 4)

for img, lab, subfig in zip(X_train, y_train, ax.reshape(-1)):
    subfig.imshow(img.reshape(28, 28), cmap="gray")
    subfig.set_xlabel(num_to_letter(lab))

plt.tight_layout()
plt.show()
print(fig)