
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree models with different depths
depths = [1, 2, 3, 5, 10, 20, None]

train_accuracy = []
test_accuracy = []

# Train and evaluate models
for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict training and testing data
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

    print("Tree Depth:", depth)
    print("Training Accuracy:", train_acc)
    print("Testing Accuracy:", test_acc)
    print("--------------------------------")

# Plot training and testing accuracy
plt.figure(figsize=(9, 5))

plt.plot(
    range(len(depths)),
    train_accuracy,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    range(len(depths)),
    test_accuracy,
    marker='s',
    label='Testing Accuracy'
)

plt.xticks(range(len(depths)), ['1', '2', '3', '5', '10', '20', 'None'])

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Analysis of Overfitting")
plt.legend()
plt.grid(True)

plt.show()

# Identify overfitting
print("\nOverfitting Analysis:")

for i, depth in enumerate(depths):
    if train_accuracy[i] > test_accuracy[i] + 0.05:
        print("Depth", depth, ": Possible Overfitting")
    else:
        print("Depth", depth, ": No Significant Overfitting")
