import pyod.utils as pyd
import matplotlib.pyplot as plt
import sklearn.metrics as skl
from pyod.models.knn import KNN
import numpy as np

#3.1

data = pyd.data.generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42)

X_train, X_test, y_train, y_test = data

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Normal', alpha=0.7)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Outliers', alpha=0.7)
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#3.2

contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.25]

print("Comparing different contamination rates:")

for contamination in contamination_rates:
    print(f"\nContamination Rate: {contamination}")
    
    knn_model = KNN(contamination=contamination)
    knn_model.fit(X_train)

    y_test_pred = knn_model.predict(X_test)
    
    print(f"Outliers: {sum(y_test_pred)}")
    
    cm = skl.confusion_matrix(y_test, y_test_pred)
    TN, FP, FN, TP = cm.ravel()
    
    balanced_acc = skl.balanced_accuracy_score(y_test, y_test_pred)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    y_test_scores = knn_model.decision_function(X_test)
    auc = skl.roc_auc_score(y_test, y_test_scores)
    
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC AUC: {auc:.4f}")

plt.figure(figsize=(10, 8))

for contamination in contamination_rates:
    knn_model = KNN(contamination=contamination)
    knn_model.fit(X_train)
    y_test_scores = knn_model.decision_function(X_test)
    
    fpr, tpr, _ = skl.roc_curve(y_test, y_test_scores)
    auc = skl.roc_auc_score(y_test, y_test_scores)
    
    plt.plot(fpr, tpr, lw=2, label=f'Contamination {contamination} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Contamination Rates')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

#3.3

data_1d = pyd.data.generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1, random_state=42)
X_train_1d, X_test_1d, y_train_1d, y_test_1d = data_1d

print(f"\nGenerated {len(X_train_1d)} training samples.")
print(f"Actual number of outliers in data: {np.sum(y_train_1d)}")

mean_1d = np.mean(X_train_1d)
std_1d = np.std(X_train_1d)
z_scores = np.abs((X_train_1d - mean_1d) / std_1d)

contamination_rate = 0.1
z_score_threshold = np.quantile(z_scores, 1 - contamination_rate)

print(f"Calculated Z-score threshold for top {contamination_rate*100}%: {z_score_threshold:.4f}")
y_pred_zscore = (z_scores > z_score_threshold).astype(int)

print(f"Number of outliers detected by Z-score method: {np.sum(y_pred_zscore)}")
balanced_acc_zscore = skl.balanced_accuracy_score(y_train_1d, y_pred_zscore)

print(f"\nBalanced Accuracy of the Z-score method: {balanced_acc_zscore:.4f}")

#3.4

n_samples = 1000
n_features = 2
contamination = 0.1
n_outliers = int(n_samples * contamination)
n_inliers = n_samples - n_outliers


mu = np.array([2, 3])
Sigma = np.array([
    [1, 0.6],
    [0.6, 2]
])

X_base = np.random.randn(n_inliers, n_features)
L = np.linalg.cholesky(Sigma)
X_inliers = (L @ X_base.T).T + mu

X_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, n_features))

X_full = np.vstack([X_inliers, X_outliers])
y_true = np.array([0] * n_inliers + [1] * n_outliers)

shuffle_idx = np.random.permutation(n_samples)
X_full = X_full[shuffle_idx]
y_true = y_true[shuffle_idx]


mu_sample = np.mean(X_full, axis=0)
Sigma_sample = np.cov(X_full, rowvar=False)
Sigma_inv = np.linalg.inv(Sigma_sample)
z_scores = np.zeros(n_samples)
for i in range(n_samples):
    diff = X_full[i] - mu_sample
    z_scores[i] = np.sqrt(diff.T @ Sigma_inv @ diff)

threshold = np.quantile(z_scores, 1 - contamination)

y_pred = (z_scores > threshold).astype(int)

bal_acc = skl.balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy of Z-score method: {bal_acc:.4f}")

plt.figure(figsize=(10, 7))
plt.scatter(X_full[y_pred == 0, 0], X_full[y_pred == 0, 1], c='blue', label='Predicted Inlier')
plt.scatter(X_full[y_pred == 1, 0], X_full[y_pred == 1, 1], c='red', label='Predicted Outlier')
plt.title('Anomaly Detection using Multi-dimensional Z-Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()