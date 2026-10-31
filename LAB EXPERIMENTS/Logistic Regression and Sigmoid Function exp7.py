
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load Breast Cancer dataset
data = load_breast_cancer()

# Features and target
X = data.data
y = data.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Logistic Regression model
model = LogisticRegression(max_iter=5000)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Results")
print("Accuracy:", accuracy * 100, "%")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Define Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate values for sigmoid function
x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)

# Plot Sigmoid Function
plt.figure(figsize=(8, 5))

plt.plot(x, y_sigmoid, color='blue',
         label='Sigmoid Function')

plt.axhline(y=0.5, color='red',
            linestyle='--', label='Threshold = 0.5')

plt.axvline(x=0, color='green',
            linestyle='--')

plt.xlabel("Input Value (x)")
plt.ylabel("Probability")
plt.title("Sigmoid Function in Logistic Regression")

plt.legend()
plt.grid(True)
plt.show()
