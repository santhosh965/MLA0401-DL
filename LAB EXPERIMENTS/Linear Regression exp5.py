
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
data = load_diabetes()

# Select one feature (BMI)
X = data.data[:, 2].reshape(-1, 1)

# Target variable
y = data.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display model coefficients
print("Linear Regression Model")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Display actual and predicted values
print("\nActual Values:")
print(y_test[:10])

print("\nPredicted Values:")
print(y_pred[:10].round(2))

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Visualize actual and predicted values
plt.figure(figsize=(8, 5))

plt.scatter(X_test, y_test, color='blue',
            label='Actual Values')

plt.plot(X_test, y_pred, color='red',
         label='Regression Line')

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression Using Diabetes Dataset")

plt.legend()
plt.grid(True)
plt.show()
