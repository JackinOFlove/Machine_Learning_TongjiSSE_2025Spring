# Machine Learning Experiment - Diabetes Dataset Linear Regression Analysis

## Experiment Objective
This experiment uses the diabetes dataset provided by scikit-learn, implements linear regression models using least squares and gradient descent methods, and evaluates and compares the two models.

## Experiment Content
1. Data preparation and preprocessing
2. Training linear regression model using least squares method
3. Training linear regression model using gradient descent method
4. Model evaluation (including custom MSE implementation)
5. Result visualization and comparison

## Dataset Description
- Data source: Scikit-learn built-in diabetes dataset
- Data content: Contains physiological data of 442 patients and their disease progression after one year
- Feature selection: For visualization purposes, we only selected Body Mass Index (BMI) as a single feature for modeling

## Code Implementation

### 1. Data Preparation
```python
# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Select single feature (BMI)
X_single = X[:, 2:3]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Model Training
```python
# Least squares linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Gradient descent linear regression
sgd_model = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, 
                        learning_rate='constant', eta0=0.01, max_iter=1000, 
                        random_state=42)
sgd_model.fit(X_train_scaled, y_train)
```

### 3. Model Evaluation
```python
# Using sklearn's MSE function
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)

# Manual implementation of MSE calculation function
def calculate_mse(y_true, y_pred):
    n = len(y_true)
    squared_errors = (y_true - y_pred) ** 2
    return np.sum(squared_errors) / n
```

### 4. Result Visualization
```python
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Test Data')
plt.plot(X_range, y_range_lr, color='blue', linewidth=2, label='Least Squares')
plt.plot(X_range, y_range_sgd, color='red', linewidth=2, label='Gradient Descent')
```

## Running Instructions
1. Environment requirements:
   - Python 3.x
   - Dependencies: numpy, matplotlib, scikit-learn

2. How to run:
   ```
   python Diabetes_Regression.py
   ```

3. Output content:
   - Console output: Dataset information, model parameters, MSE values and performance comparison results
   - Image output: `Diabetes_Regression_Comparison.png`

## Experiment Result Analysis
The program calculates and displays the mean squared error of both models, and determines which model performs better by comparison. The image clearly shows the fitting effects of both models on the test set, and the visualization makes the differences between the two methods readily apparent.

## Extended Content
1. Implementation of custom MSE calculation function, verifying consistency with sklearn library function results
2. Visualization of fitting effects of both models, providing an intuitive comparison of the differences between the two methods 