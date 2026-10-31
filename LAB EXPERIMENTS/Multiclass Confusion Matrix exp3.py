
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Actual class values
actual = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

# Predicted class values
predicted = [0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2]

# Define class names
class_names = ['Class A', 'Class B', 'Class C']

# Generate confusion matrix
cm = confusion_matrix(actual, predicted)

# Display confusion matrix
print("Multiclass Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(actual, predicted)
print("\nAccuracy:", accuracy * 100, "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    target_names=class_names
))

# Visualize confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Multiclass Confusion Matrix")
plt.show()
