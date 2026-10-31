# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Actual values
y_actual = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted values
y_predicted = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Calculate confusion matrix
cm = confusion_matrix(y_actual, y_predicted)

# Display confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_actual, y_predicted)
print("\nAccuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_actual, y_predicted))

# Visualize confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])

plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')

plt.show()