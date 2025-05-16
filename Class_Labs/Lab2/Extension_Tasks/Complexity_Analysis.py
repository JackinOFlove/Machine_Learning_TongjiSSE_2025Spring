import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                           precision_score, recall_score, f1_score, accuracy_score)

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """
    Load the digits dataset and split into training and test sets
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, digits.target_names

def analyze_model_complexity(X_train, X_test, y_train, y_test):
    """
    Analyze how model complexity (gamma in SVM) affects bias and variance
    """
    # Define a range of gamma values to test
    gamma_range = np.logspace(-5, 4, 20)
    
    # Initialize arrays to store results
    train_scores = []
    test_scores = []
    train_loss = []
    test_loss = []
    bias_values = []
    variance_values = []
    
    for gamma in gamma_range:
        # Train SVM with current gamma
        svm = SVC(kernel='rbf', gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)
        
        # Calculate training and test scores (accuracy)
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        # Store accuracy scores
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        # Calculate loss (1 - accuracy)
        train_loss.append(1 - train_score)
        test_loss.append(1 - test_score)
        
        # Calculate bias and variance components
        # Bias is approximated by training error
        bias = 1 - train_score
        
        # Variance is approximated by the difference between test and training error
        variance = abs((1 - test_score) - (1 - train_score))
        
        bias_values.append(bias)
        variance_values.append(variance)
    
    # Plot the results
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Loss curve
    plt.subplot(2, 2, 1)
    plt.semilogx(gamma_range, train_loss, 'b-', label='train loss')
    plt.semilogx(gamma_range, test_loss, 'r-', label='test loss')
    plt.xlabel('gamma')
    plt.ylabel('Loss')
    plt.title('SVM_LOSS')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Bias-Variance curve
    plt.subplot(2, 2, 2)
    plt.semilogx(gamma_range, bias_values, 'r-', label='test bias')
    plt.semilogx(gamma_range, variance_values, 'g-', label='test variance')
    plt.semilogx(gamma_range, test_loss, 'b-', label='test loss')
    plt.xlabel('gamma')
    plt.ylabel('error')
    plt.title('SVM Bias and Variance')
    plt.legend()
    plt.grid(True)
    
    # Return the best gamma value based on test score
    best_idx = np.argmax(test_scores)
    best_gamma = gamma_range[best_idx]
    best_test_score = test_scores[best_idx]
    
    print(f"Best gamma value: {best_gamma}")
    print(f"Best test score: {best_test_score}")
    
    return best_gamma

def k_fold_cross_validation(X, y, gamma_values):
    """
    Perform k-fold cross-validation to find the best gamma value
    """
    cv_scores = []
    
    for gamma in gamma_values:
        svm = SVC(kernel='rbf', gamma=gamma, random_state=42)
        scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Plot cross-validation results
    plt.subplot(2, 2, (3, 4))  # Use bottom half of the figure
    plt.semilogx(gamma_values, cv_scores, 'g-o')
    plt.xlabel('gamma')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('5-Fold Cross-Validation Scores for Different Gamma Values')
    plt.grid(True)
    
    # Return the best gamma value
    best_idx = np.argmax(cv_scores)
    best_gamma = gamma_values[best_idx]
    
    print(f"Best gamma from cross-validation: {best_gamma}")
    print(f"Best cross-validation score: {cv_scores[best_idx]}")
    
    return best_gamma

def plot_pr_and_roc_curves(X_train, X_test, y_train, y_test, gamma):
    """
    Plot Precision-Recall and ROC curves for each class in the dataset
    """
    # Train SVM with the best gamma
    svm = SVC(kernel='rbf', gamma=gamma, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Get probabilities for each class
    y_prob = svm.predict_proba(X_test)
    
    # Create a new figure with wider aspect ratio
    plt.figure(figsize=(24, 10))
    
    # For storing average precision and ROC AUC
    avg_precision = []
    avg_auc = []
    
    # Get number of classes
    n_classes = len(np.unique(y_test))
    
    # Subplot for Precision-Recall curve
    plt.subplot(1, 2, 1)
    
    # For each class
    for i in range(n_classes):
        # Binarize the output
        y_test_bin = (y_test == i).astype(int)
        
        # Calculate precision and recall
        precision, recall, _ = precision_recall_curve(y_test_bin, y_prob[:, i])
        
        # Plot PR curve
        plt.plot(recall, precision, lw=2, 
                 label=f'class {i}')
        
        # Calculate average precision
        avg_precision.append(precision_score(y_test_bin, (y_prob[:, i] > 0.5).astype(int)))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)
    
    # Subplot for ROC curve
    plt.subplot(1, 2, 2)
    
    # For each class
    for i in range(n_classes):
        # Binarize the output
        y_test_bin = (y_test == i).astype(int)
        
        # Calculate false positive rate and true positive rate
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob[:, i])
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        avg_auc.append(roc_auc)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, 
                 label=f'class {i}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="best")
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('PR_ROC_Curves.png')
    plt.show()
    
    print(f"Average Precision across all classes: {np.mean(avg_precision):.4f}")
    print(f"Average AUC across all classes: {np.mean(avg_auc):.4f}")

def main():
    # Load data
    print("Loading digits dataset...")
    X_train, X_test, y_train, y_test, target_names = load_data()
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Analyze model complexity
    print("\nAnalyzing model complexity...")
    best_gamma = analyze_model_complexity(X_train, X_test, y_train, y_test)
    
    # Perform k-fold cross-validation
    print("\nPerforming k-fold cross-validation...")
    gamma_values = np.logspace(-5, 4, 20)
    cv_best_gamma = k_fold_cross_validation(X_train, y_train, gamma_values)
    
    # Save the complexity analysis plot
    plt.tight_layout()
    plt.savefig('Complexity_Analysis.png')
    plt.show()
    
    # Plot PR and ROC curves
    print("\nPlotting PR and ROC curves...")
    plot_pr_and_roc_curves(X_train, X_test, y_train, y_test, cv_best_gamma)
    
    print("\nAnalysis completed. Results saved to Complexity_Analysis.png and PR_ROC_Curves.png")

if __name__ == "__main__":
    main() 