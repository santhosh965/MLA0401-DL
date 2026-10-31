
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Actual class values
actual = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Predicted class values
predicted = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# Generate confusion matrix
cm = confusion_matrix(actual, predicted)

# Display confusion matrix
print("Two-Class Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(actual, predicted)
print("\nAccuracy:", accuracy * 100, "%")

# Extract TP, TN, FP, FN
TN, FP, FN, TP = cm.ravel()

print("\nTrue Positive (TP):", TP)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Two-Class Confusion Matrix")

plt.show()
