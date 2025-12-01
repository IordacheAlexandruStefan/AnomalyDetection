import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Data Generation
n_train = 300
n_test = 200
n_features = 3
contamination = 0.15

X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train, 
    n_test=n_test, 
    n_features=n_features, 
    contamination=contamination, 
    random_state=42
)

def visualize_results(X_train, y_train, pred_train, X_test, y_test, pred_test, title):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    def plot_3d(ax, X, y, sub_title):
        ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='b', label='Inliers', s=20, edgecolors='k')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='r', label='Outliers', s=20, edgecolors='k')
        ax.set_title(sub_title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()

    # 1. Train Data (Ground Truth)
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d(ax1, X_train, y_train, "Training Data (Ground Truth)")

    # 2. Train Data (Predicted)
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d(ax2, X_train, pred_train, "Training Data (Predicted)")

    # 3. Test Data (Ground Truth)
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d(ax3, X_test, y_test, "Test Data (Ground Truth)")

    # 4. Test Data (Predicted)
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d(ax4, X_test, pred_test, "Test Data (Predicted)")

    plt.tight_layout()
    plt.show()

# OCSVM (Linear Kernel)
print("-" * 30)
print("Running OCSVM (Linear)...")
ocsvm_linear = OCSVM(kernel='linear', contamination=contamination)
ocsvm_linear.fit(X_train)

y_train_pred_lin = ocsvm_linear.predict(X_train)
y_test_pred_lin = ocsvm_linear.predict(X_test)
y_test_scores_lin = ocsvm_linear.decision_function(X_test)

ba_lin = balanced_accuracy_score(y_test, y_test_pred_lin)
auc_lin = roc_auc_score(y_test, y_test_scores_lin)

print(f"OCSVM (Linear) - Balanced Accuracy: {ba_lin:.4f}")
print(f"OCSVM (Linear) - ROC AUC: {auc_lin:.4f}")

visualize_results(X_train, y_train, y_train_pred_lin, 
                  X_test, y_test, y_test_pred_lin, 
                  "OCSVM (Linear Kernel)")

# OCSVM (RBF Kernel)
print("-" * 30)
print("Running OCSVM (RBF)...")
ocsvm_rbf = OCSVM(kernel='rbf', contamination=contamination)
ocsvm_rbf.fit(X_train)

y_train_pred_rbf = ocsvm_rbf.predict(X_train)
y_test_pred_rbf = ocsvm_rbf.predict(X_test)
y_test_scores_rbf = ocsvm_rbf.decision_function(X_test)

ba_rbf = balanced_accuracy_score(y_test, y_test_pred_rbf)
auc_rbf = roc_auc_score(y_test, y_test_scores_rbf)

print(f"OCSVM (RBF) - Balanced Accuracy: {ba_rbf:.4f}")
print(f"OCSVM (RBF) - ROC AUC: {auc_rbf:.4f}")

visualize_results(X_train, y_train, y_train_pred_rbf, 
                  X_test, y_test, y_test_pred_rbf, 
                  "OCSVM (RBF Kernel)")

# DeepSVDD
print("-" * 30)
print("Running DeepSVDD...")

deep_svdd = DeepSVDD(n_features=n_features, contamination=contamination, epochs=20, random_state=42) 
deep_svdd.fit(X_train)

y_train_pred_deep = deep_svdd.predict(X_train)
y_test_pred_deep = deep_svdd.predict(X_test)
y_test_scores_deep = deep_svdd.decision_function(X_test)

ba_deep = balanced_accuracy_score(y_test, y_test_pred_deep)
auc_deep = roc_auc_score(y_test, y_test_scores_deep)

print(f"DeepSVDD - Balanced Accuracy: {ba_deep:.4f}")
print(f"DeepSVDD - ROC AUC: {auc_deep:.4f}")

visualize_results(X_train, y_train, y_train_pred_deep, 
                  X_test, y_test, y_test_pred_deep, 
                  "DeepSVDD")

import scipy.io
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, make_scorer

# Load Data
try:
    data = scipy.io.loadmat('cardio.mat')
    X = data['X']
    y = data['y'].ravel() # Flatten to 1D array
except FileNotFoundError:
    exit()

print(f"Dataset shape: {X.shape}")
print(f"Original label distribution: {np.unique(y, return_counts=True)}")

# Label Conversion (Crucial Step)
y_sklearn = np.where(y == 0, 1, -1)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_sklearn, 
    train_size=0.40, 
    stratify=y_sklearn, 
    random_state=42
)

n_outliers = np.sum(y_train == -1)
contamination_rate = n_outliers / len(y_train)
print(f"Calculated Contamination Rate: {contamination_rate:.4f}")

# Pipeline & Grid Search Setup
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])

param_grid = {
    'ocsvm__kernel': ['rbf', 'sigmoid', 'poly', 'linear'],
    'ocsvm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'ocsvm__nu': [0.01, 0.05, 0.1, 0.2, contamination_rate]
}
scorer = make_scorer(balanced_accuracy_score)


print("Starting GridSearchCV... (this may take a moment)")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,            
    n_jobs=-1,       
    verbose=1
)

# Training
grid_search.fit(X_train, y_train)

# Results & Evaluation
print("\n" + "="*40)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Balanced Accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

test_ba = balanced_accuracy_score(y_test, y_test_pred)

print(f"Test Set Balanced Accuracy: {test_ba:.4f}")
print("="*40)