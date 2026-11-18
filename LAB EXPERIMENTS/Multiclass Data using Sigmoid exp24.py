
# EXPERIMENT 24: Multiclass Data using Sigmoid

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate multiclass dataset
X, y = make_blobs(
    n_samples=1200,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

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
    tf.keras.layers.Dense(3, activation='softmax')
])

# 5. Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train model
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    verbose=0
)

# 7. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Experiment 24: Multiclass Data using Sigmoid")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# 8. Predict classes
predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Plot accuracy and loss
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

# 10. Plot decision boundary
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5, 200),
    np.linspace(X_train[:, 1].min()-0.5, X_train[:, 1].max()+0.5, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
predictions = model.predict(grid, verbose=0)
Z = np.argmax(predictions, axis=1).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
plt.title('Multiclass Data - Sigmoid Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
