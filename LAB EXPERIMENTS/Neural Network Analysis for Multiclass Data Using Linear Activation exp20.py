import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Generate multiclass data
X, y = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42,
    cluster_std=1.5
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(2, 2),
    activation='identity',
    learning_rate_init=0.01,
    solver='sgd',
    max_iter=2000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Plot multiclass data
plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap='viridis',
    edgecolors='black'
)

plt.title("Multiclass Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()