import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Data Preparation
print("Loading diabetes dataset...")
# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Print dataset information
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {diabetes.feature_names}")
print(f"Target range: [{y.min()}, {y.max()}]")

# For intuitive visualization, we only select one feature (BMI)
X_single = X[:, 2:3]  # BMI feature
print(f"Selected feature: {diabetes.feature_names[2]}")

# Split dataset into training and test sets, set random seed to 42
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Standardize data (important for gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Model Training
print("\nTraining models...")

# 2.1 Linear Regression using Least Squares
print("Training Least Squares model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print(f"Least Squares model coefficient: {lr_model.coef_}")
print(f"Least Squares model intercept: {lr_model.intercept_}")

# 2.2 Linear Regression using Gradient Descent
print("Training Gradient Descent model...")
sgd_model = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, 
                        learning_rate='constant', eta0=0.01, max_iter=1000, 
                        random_state=42)
sgd_model.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_model.predict(X_test_scaled)

print(f"Gradient Descent model coefficient: {sgd_model.coef_}")
print(f"Gradient Descent model intercept: {sgd_model.intercept_}")

# 3. Model Evaluation
print("\nEvaluating models...")

# 3.1 Using sklearn's mean squared error function
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
print(f"Least Squares MSE calculated by sklearn: {mse_lr:.2f}")
print(f"Gradient Descent MSE calculated by sklearn: {mse_sgd:.2f}")

# 3.2 Manual implementation of MSE calculation function (extended requirement)
def calculate_mse(y_true, y_pred):
    """
    Manually calculate mean squared error
    MSE = (1/n) * Σ(y_true - y_pred)^2
    """
    n = len(y_true)
    squared_errors = (y_true - y_pred) ** 2
    return np.sum(squared_errors) / n

# Use custom function to calculate MSE
manual_mse_lr = calculate_mse(y_test, y_pred_lr)
manual_mse_sgd = calculate_mse(y_test, y_pred_sgd)
print(f"Least Squares MSE calculated manually: {manual_mse_lr:.2f}")
print(f"Gradient Descent MSE calculated manually: {manual_mse_sgd:.2f}")

# Verify if manually calculated MSE matches sklearn's results
print(f"Least Squares MSE calculation difference: {abs(mse_lr - manual_mse_lr):.10f}")
print(f"Gradient Descent MSE calculation difference: {abs(mse_sgd - manual_mse_sgd):.10f}")

# 4. Visualize Results (extended requirement)
print("\nVisualizing results...")

plt.figure(figsize=(12, 8))

# 4.1 Draw scatter plot
plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Test Data')

# 4.2 To draw smooth lines, we create a data range from min to max
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)

# 4.3 Predict values for both models
y_range_lr = lr_model.predict(X_range)
y_range_sgd = sgd_model.predict(X_range_scaled)

# 4.4 Draw fitting lines for both models
plt.plot(X_range, y_range_lr, color='blue', linewidth=2, label='Least Squares')
plt.plot(X_range, y_range_sgd, color='red', linewidth=2, label='Gradient Descent')

# 4.5 Set chart properties
plt.title('Diabetes Dataset Regression Model Comparison (BMI vs Disease Progression)')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Disease Progression Indicator')
plt.legend()
plt.grid(True)

# 4.6 Save image
plt.savefig('Diabetes_Regression_Comparison.png')
print("Image saved as 'Diabetes_Regression_Comparison.png'")

# 4.7 Display image
plt.show()

# 5. Compare which model is better
print("\nModel comparison:")
if mse_lr < mse_sgd:
    print("Least Squares model performs better")
else:
    print("Gradient Descent model performs better")

print("\nExperiment completed!")