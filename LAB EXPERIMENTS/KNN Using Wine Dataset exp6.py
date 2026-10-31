
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load Wine dataset
wine = load_wine()

# Features and target
X = wine.data
y = wine.target

# Display dataset information
print("Wine Dataset")
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Wine Classes:", wine.target_names)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nKNN Classification Results")
print("Accuracy:", accuracy * 100, "%")

# Display actual and predicted values
print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=wine.target_names
))

# Visualize confusion matrix
plt.figure(figsize=(6, 4))

plt.imshow(cm, cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1, 2], wine.target_names)
plt.yticks([0, 1, 2], wine.target_names)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j],
                 ha='center', va='center',
                 color='black')

plt.colorbar()
plt.show()
