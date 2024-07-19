import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

X, Y = [], []
for i in range(5):
    tmp = np.load(f"training_data/{i}.npy")
    X.append(tmp)
    for _ in range(len(tmp)):
        Y.append(i)

X = np.row_stack(X).reshape(-1, 21*2*20)
Y = np.array(Y)

global_scaller = StandardScaler()
X = global_scaller.fit_transform(X)

scaller = StandardScaler()
X = scaller.fit_transform(X)


reducer = PCA(n_components=2)
reducer = LinearDiscriminantAnalysis(n_components=2)
X_reduced = reducer.fit_transform(X, Y)
names = ["Swipe", "Choose", "Pick", "Something", "Something^2"]

colors = ["red", "yellow", "blue", "black", "pink"]

fig = plt.figure("X")
ax = fig.add_subplot(1, 1, 1)
for i in range(1, 6):
    ax.scatter(X_reduced[Y==i, 0], X_reduced[Y==i, 1], label=names[i-1], color=colors[i-1])

ax.legend()

plt.show()