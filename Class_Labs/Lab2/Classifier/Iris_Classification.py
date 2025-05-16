import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Load Iris dataset
def load_data():
    """
    Load the Iris dataset and split into training and test sets
    Returns:
        X_train, X_test, y_train, y_test: Split training and test sets
        feature_names: Names of the features
        target_names: Names of the target classes
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Make the classification task more challenging by using a smaller training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

def Predict(classifier_name, X_train, y_train, X_test):
    """
    Train a classifier and make predictions based on the specified classifier name
    
    Args:
        classifier_name: Name of the classifier ('KNN', 'Logistic', 'DecisionTree', 'SVM')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
    
    Returns:
        y_pred: Predicted labels
        classifier: Trained classifier
    """
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'Logistic':
        classifier = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', solver='lbfgs')
    elif classifier_name == 'DecisionTree':
        classifier = DecisionTreeClassifier(random_state=42, max_depth=3)  # Limit depth to avoid overfitting
    elif classifier_name == 'SVM':
        classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    else:
        raise ValueError(f"Unsupported classifier name: {classifier_name}")
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    return y_pred, classifier

def evaluate_classifier(y_test, y_pred, classifier_name, X_train, y_train, classifier):
    """
    Evaluate classifier performance
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        classifier_name: Name of the classifier
        X_train: Training features
        y_train: Training labels
        classifier: Trained classifier object
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For multi-class problems, calculate precision, recall, and F1 with different averaging methods
    # Micro average (aggregates the contributions of all classes)
    precision_micro = precision_score(y_test, y_pred, average='micro')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    
    # Macro average (equal weight for each class)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Weighted average (weighted by class support)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Perform cross-validation for more reliable evaluation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    
    # Print evaluation results
    print(f"\n{classifier_name} Classifier Evaluation:")
    print(f"Accuracy: {accuracy:.6f}")
    
    print("\nMetrics for Different Averaging Methods:")
    
    print("\n1. Micro Average (Aggregate contributions of all classes):")
    print(f"  Precision: {precision_micro:.6f}")
    print(f"  Recall: {recall_micro:.6f}")
    print(f"  F1 Score: {f1_micro:.6f}")
    print(f"  Note: For balanced datasets, micro average precision/recall/F1 equals accuracy.")
    
    print("\n2. Macro Average (Equal weight for each class):")
    print(f"  Precision: {precision_macro:.6f}")
    print(f"  Recall: {recall_macro:.6f}")
    print(f"  F1 Score: {f1_macro:.6f}")
    
    print("\n3. Weighted Average (Weighted by class support):")
    print(f"  Precision: {precision_weighted:.6f}")
    print(f"  Recall: {recall_weighted:.6f}")
    print(f"  F1 Score: {f1_weighted:.6f}")
    
    print(f"\nCross-validation Accuracy: {cv_scores.mean():.6f} (±{cv_scores.std():.6f})")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=6))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate per-class metrics
    print("\nPer-class Metrics:")
    for i in range(3):  # 3 classes in Iris dataset
        class_indices = [idx for idx, val in enumerate(y_test) if val == i]
        if class_indices:
            class_y_test = [y_test[idx] for idx in class_indices]
            class_y_pred = [y_pred[idx] for idx in class_indices]
            
            class_precision = precision_score(class_y_test, class_y_pred, average='binary', pos_label=i)
            class_recall = recall_score(class_y_test, class_y_pred, average='binary', pos_label=i)
            class_f1 = f1_score(class_y_test, class_y_pred, average='binary', pos_label=i)
            
            print(f"Class {i}:")
            print(f"  Precision: {class_precision:.6f}")
            print(f"  Recall: {class_recall:.6f}")
            print(f"  F1 Score: {class_f1:.6f}")

def Vis(X, y, classifier, target_names, classifier_name):
    """
    Visualize classification results
    
    Args:
        X: Feature data
        y: True labels
        classifier: Trained classifier
        target_names: Names of target classes
        classifier_name: Name of the classifier
    """
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a mesh grid to plot decision boundaries
    h = 0.02  # Step size
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # For better decision boundary visualization, train a new classifier on PCA-transformed data
    # This ensures the decision boundaries in 2D space are accurate
    pca_classifier = None
    if classifier_name == 'KNN':
        pca_classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'Logistic':
        pca_classifier = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', solver='lbfgs')
    elif classifier_name == 'DecisionTree':
        pca_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
    elif classifier_name == 'SVM':
        pca_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    
    # Train on PCA-transformed data
    pca_classifier.fit(X_pca, y)
    
    # Predict directly on the mesh grid points in PCA space
    Z = pca_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Get predictions for the test data using the PCA-trained classifier
    y_pred = pca_classifier.predict(X_pca)
    
    # Plot decision boundary
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Define markers and colors
    colors = ['navy', 'turquoise', 'darkorange']
    true_markers = ['x', 'x', 'x']  # Use 'x' to mark true values
    pred_markers = ['o', 'o', 'o']  # Use 'o' to mark predicted values
    
    # Plot each class
    for i in range(3):
        # Plot points predicted as class i
        idx_pred = np.where(y_pred == i)
        plt.scatter(X_pca[idx_pred, 0], X_pca[idx_pred, 1], c=colors[i], 
                    marker=pred_markers[i], s=40, alpha=0.8,
                    label=f'pred {i} ({target_names[i]})')
        
        # Plot points truly belonging to class i
        idx_true = np.where(y == i)
        plt.scatter(X_pca[idx_true, 0], X_pca[idx_true, 1], c=colors[i], 
                    marker=true_markers[i], s=80, alpha=0.8, edgecolors='black',
                    label=f'true {i} ({target_names[i]})')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Iris Classification: {classifier_name} Classifier')
    plt.legend(loc='best')
    plt.savefig(f'Iris_Classification_{classifier_name}.png')
    plt.show()

def main():
    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    # Print dataset information
    print("Iris Dataset Information:")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Number of classes: {len(target_names)}")
    print(f"Class names: {target_names}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # List of classifier names
    classifier_names = ['KNN', 'Logistic', 'DecisionTree', 'SVM']
    
    # Train, predict, and evaluate each classifier
    for name in classifier_names:
        # Train classifier and predict
        y_pred, classifier = Predict(name, X_train, y_train, X_test)
        
        # Evaluate classifier
        evaluate_classifier(y_test, y_pred, name, X_train, y_train, classifier)
        
        # Visualize classification results
        print(f"\nGenerating visualization for {name} classifier...")
        Vis(X_test, y_test, classifier, target_names, name)

if __name__ == "__main__":
    main() 