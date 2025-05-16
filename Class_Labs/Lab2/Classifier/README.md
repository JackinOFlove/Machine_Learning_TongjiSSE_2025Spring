# Iris Dataset Classification Experiment

## Overview
This experiment implements and compares multiple classification algorithms on the Iris dataset using scikit-learn. It evaluates the performance of KNN, Logistic Regression, Decision Tree, and SVM classifiers through various metrics and provides visualization of decision boundaries.

## Dataset Description
- **Source**: Scikit-learn's built-in Iris dataset
- **Content**: Contains 150 samples of iris flowers from three different species (50 each)
- **Features**: Each sample has 4 features - sepal length, sepal width, petal length, and petal width
- **Classes**: Setosa, Versicolor, and Virginica

## Implementation Details

### 1. Data Processing
- Loading the Iris dataset
- Splitting into training (60%) and test (40%) sets
- Standardizing features for better model performance

### 2. Classification Models
The experiment implements four different classifiers:
- **K-Nearest Neighbors (KNN)**: Using 5 neighbors
- **Logistic Regression**: Multinomial with lbfgs solver
- **Decision Tree**: With limited maximum depth to prevent overfitting
- **Support Vector Machine (SVM)**: Using RBF kernel

### 3. Evaluation Metrics
For thorough evaluation, the following metrics are calculated with 6 decimal precision:
- **Accuracy**: Percentage of correctly classified samples
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall

For multi-class problems, three averaging methods are used:

#### Micro Average
- Aggregates contributions of all classes by counting total TP, FP, FN
- Best for imbalanced datasets
- For balanced datasets, micro-average equals accuracy
- Formula: Calculate metrics globally after combining all class predictions

#### Macro Average
- Simple average across all classes (equal weight)
- Treats all classes equally regardless of their support
- Can be misleading for imbalanced datasets
- Formula: (metric_class1 + metric_class2 + ... + metric_classN) / N

#### Weighted Average
- Weighted by class frequency/support
- Takes class imbalance into account
- Accounts for each class's proportion in the evaluation
- Formula: (metric_class1 * support_class1 + ... + metric_classN * support_classN) / total_support

Additional evaluation measures:
- **Cross-validation**: 5-fold cross-validation to assess model stability
- **Per-class metrics**: Individual precision, recall, and F1 scores for each class
- **Confusion Matrix**: Table showing prediction counts for each class combination

### 4. Visualization
- **Dimensionality Reduction**: Using PCA to reduce the 4D feature space to 2D
- **Decision Boundaries**: Visualizing how each classifier separates the classes
- **Class Distribution**: Plotting samples from different classes with distinct colors
- **Model Identification**: Each visualization clearly indicates which classifier was used

## Running the Code
```
python Iris_Classification.py
```

## Output
- Comprehensive dataset information
- Detailed evaluation metrics for each classifier with all three averaging methods
- Visual representation of classification boundaries
- PNG images saved for each classifier's visualization

## Analysis
The experiment provides insights into:
1. The comparative performance of different classification algorithms on the Iris dataset
2. How different averaging methods affect the interpretation of results in multi-class problems
3. Visual understanding of decision boundaries and how they differ between classifiers

## Reference to Assignment 1
The implementation follows the concepts from Assignment 1 regarding multi-class metrics:
- **Binary Metrics**: Applied to each class separately in a one-vs-rest approach
- **Micro Average**: Aggregates metrics globally
- **Macro Average**: Treats all classes equally
- **Weighted Average**: Adjusts for class imbalance

## Extensions
1. Parameter tuning to optimize each classifier
2. Implementation of additional classifiers (Random Forest, Gradient Boosting, etc.)
3. Feature importance analysis
4. Interactive confusion matrix visualization 