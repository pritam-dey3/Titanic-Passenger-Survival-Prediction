import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def modified_confusion_matrix(y, y_hat):
    conf = confusion_matrix(y, y_hat)
    D = np.diag(pd.value_counts(y).sort_index())
    conf_mod = np.dot(np.linalg.inv(D), conf)
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.imshow(conf_mod, cmap=plt.get_cmap('jet'))
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.grid(False)