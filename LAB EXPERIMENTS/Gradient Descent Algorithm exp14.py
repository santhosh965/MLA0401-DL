# Import required library
import numpy as np
import matplotlib.pyplot as plt

# Define dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Initialize parameters
m = 0
c = 0

# Learning rate
learning_rate = 0.01

# Number of iterations
epochs = 1000

n = len(X)

# Gradient Descent Algorithm
for i in range(epochs):

    # Predict output
    y_pred = m * X + c

    # Calculate errors
    error = y_pred - y

    # Calculate gradients
    dm = (2/n) * np.sum(error * X)
    dc = (2/n) * np.sum(error)

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

# Display results
print("Slope (m):", m)
print("Intercept (c):", c)

# Predict output
y_pred = m * X + c

print("Predicted Values:", y_pred)

# Plot graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color='red', label="Predicted Line")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Gradient Descent Algorithm")
plt.legend()
plt.show()