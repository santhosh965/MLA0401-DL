import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate data
X, y = make_blobs(
    n_samples=100,
    centers=2,
    random_state=6,
    cluster_std=1.0
)

# Train linear SVM
model = SVC(kernel='linear')
model.fit(X, y)

# Create mesh grid
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# Predict
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap='coolwarm',
    edgecolors='black'
)

plt.title("Linear Separability")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Accuracy:", model.score(X, y))