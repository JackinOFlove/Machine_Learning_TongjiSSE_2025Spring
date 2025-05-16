# Model Complexity Analysis Extension

## Overview
This extension task explores the relationship between model complexity, bias, and variance in machine learning classifiers. The code specifically focuses on Support Vector Machines (SVMs) and how the gamma parameter affects their performance on a larger dataset (handwritten digits).

## Implementation

### 1. Complexity Analysis with SVM
- **Parameter of Interest**: Gamma in SVM with RBF kernel
  - Low gamma → simpler model (higher bias, lower variance)
  - High gamma → more complex model (lower bias, higher variance)
- **Datasets Used**: `sklearn.datasets.load_digits()` (handwritten digit recognition)
- **Metrics Tracked**:
  - Training and testing loss
  - Bias (approximated by training error)
  - Variance (approximated by test-train error difference)

### 2. K-fold Cross Validation
- Implements 5-fold cross-validation to select the optimal gamma value
- Plots cross-validation scores across different gamma values
- Identifies the gamma that produces the highest average performance

### 3. Visualization
The code produces several key visualizations:

#### Complexity Analysis Charts
1. **SVM Loss Plot**: Shows how training and testing loss change with different gamma values, visualizing the underfitting and overfitting regions
2. **Bias-Variance Plot**: Illustrates the trade-off between bias and variance as model complexity increases

#### Performance Evaluation Charts
1. **Precision-Recall Curves**: For each class in the dataset
   - Shows the trade-off between precision and recall
   - Useful for evaluating performance on imbalanced datasets

2. **ROC Curves**: For each class in the dataset
   - Plots True Positive Rate vs. False Positive Rate
   - Includes Area Under the Curve (AUC) calculation

## Running the Code
```bash
python complexity_analysis.py
```

## Output
The script will generate:
1. Console output with:
   - Dataset information
   - Best gamma values found through different methods
   - Performance metrics for the optimal model

2. Image files:
   - `Complexity_Analysis.png`: Contains the loss and bias-variance plots
   - `PR_ROC_Curves.png`: Contains the precision-recall and ROC curves

## Interpretation of Results
- **U-shaped test loss curve**: Indicates the optimal complexity (gamma value) for the model
- **Bias-variance trade-off**: Shows how increasing complexity reduces bias but increases variance
- **PR and ROC curves**: Demonstrate model performance across different classification thresholds

## Extension Ideas
- Apply similar analysis to other classifiers (Random Forest, Neural Networks)
- Explore other complexity parameters (C in SVM, depth in decision trees)
- Implement learning curves to analyze how performance changes with training set size 