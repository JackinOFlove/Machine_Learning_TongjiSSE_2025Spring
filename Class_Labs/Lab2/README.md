# Machine Learning Lab 2 - Regression and Classification

## Overview
This lab implements and compares machine learning algorithms for regression and classification tasks. The experiment consists of three parts: regression analysis, classification analysis, and extension tasks.

## 1. Regression Analysis

### Dataset
- **Diabetes Dataset**: Scikit-learn's built-in diabetes dataset with 442 patients

### Algorithms
1. **Least Squares Linear Regression**: Direct solution using closed-form equation
2. **Gradient Descent Linear Regression**: Iterative optimization with tuned parameters

### Evaluation
- **Mean Squared Error (MSE)** for regression performance
- **Classification Metrics**: Accuracy, precision, recall, and F1 score

### Visualization
- Scatter plots with regression lines
- Comparison of evaluation metrics

## 2. Classification Analysis

### Dataset
- **Iris Dataset**: 3 classes of iris flowers with 4 features

### Algorithms
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Decision Tree**
4. **Support Vector Machine (SVM)**

### Evaluation
- **Accuracy, Precision, Recall, F1 Score**
- Three averaging methods: Micro, Macro, and Weighted

### Visualization
- PCA dimensionality reduction (4D to 2D)
- Decision boundary visualization
- Comparison of true vs predicted classes

## 3. Extension Tasks

### Model Complexity Analysis
- Study of SVM's gamma parameter on bias-variance tradeoff
- Using handwritten digits dataset

### Cross-Validation
- K-fold cross-validation for parameter selection
- Comparison across different complexity levels

### Advanced Evaluation
- PR Curve (Precision-Recall)
- ROC Curve and AUC values

## Key Findings

1. **Regression Comparison**:
   - Performance tradeoffs between least squares and gradient descent
   - Parameter sensitivity in gradient descent

2. **Classifier Performance**:
   - SVM typically performs best on the iris dataset
   - Different averaging methods affect evaluation metrics

3. **Bias-Variance Tradeoff**:
   - Complexity increases: bias decreases, variance increases
   - Optimal complexity point exists

## Code Structure
```
Lab2/
├── Regression/
│   ├── Diabetes_Regression.py
│   ├── Diabetes_Regression_Comparison.png
│   └── README.md
├── Classifier/
│   ├── Iris_Classification.py
│   ├── Iris_Classification_*.png
│   └── README.md
└── Extension_Tasks/
    ├── Complexity_Analysis.py
    ├── Complexity_Analysis.png
    ├── PR_ROC_Curves.png
    └── README.md
```