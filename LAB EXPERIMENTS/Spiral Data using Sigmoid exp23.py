
# EXPERIMENT 23: Spiral Data using Sigmoid

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate spiral dataset
np.random.seed(42)

n = 600
theta = np.linspace(0, 4 * np.pi, n)
r = np.linspace(0.1, 1, n)

# First spiral
x1 = r * np.cos(theta) + np.random.normal(0, 0.05, n)
y1 = r * np.sin(theta) + np.random.normal(0, 0.05, n)

# Second spiral
x2 = -r * np.cos(theta) + np.random.normal(0, 0.05, n)
y2 = -r * np.sin(theta) + np.random.normal(0, 0.05, n)

X = np.vstack([
    np.column_stack((x1, y1)),
    np.column_stack((x2, y2))
])

y = np.hstack([
    np.zeros(n),
    np.ones(n)
])

# Shuffle dataset
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Train model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    verbose=0
)

# 7. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Experiment 23: Spiral Data using Sigmoid")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# 8. Plot accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# 9. Plot decision boundary
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5, 200),
    np.linspace(X_train[:, 1].min()-0.5, X_train[:, 1].max()+0.5, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
predictions = model.predict(grid, verbose=0)
Z = (predictions > 0.5).astype(int).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
plt.title('Spiral Data - Sigmoid Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
