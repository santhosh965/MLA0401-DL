import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Generate circular data
X, y = make_circles(
    n_samples=300,
    noise=0.08,
    factor=0.4,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    learning_rate_init=0.03,
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

# Plot circular data
plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap='coolwarm',
    edgecolors='black'
)

plt.title("Circular Data Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()