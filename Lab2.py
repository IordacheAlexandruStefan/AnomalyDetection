import pyod.utils as pyd
import matplotlib.pyplot as plt
import sklearn.metrics as skl
from pyod.models.knn import KNN
import numpy as np
from pyod.utils.data import generate_data_clusters
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF


a, b = 1, 5
n_regular = 50
n_special = 10
noise_variances = [0.5, 2.0, 10.0]

regular_x_scale = 1.5 
leverage_x_range = [7, 9] 
outlier_y_error_range = [15, 25] 

fig, axes = plt.subplots(1, len(noise_variances), figsize=(18, 6), sharex=True, sharey=True)
fig.suptitle('Leverage Scores for Different Noise Variances (σ²)', fontsize=16)

for i, variance in enumerate(noise_variances):
    ax = axes[i]  
    #Regular point
    x_regular = np.random.normal(loc=0, scale=regular_x_scale, size=n_regular)
    noise_regular = np.random.normal(loc=0, scale=np.sqrt(variance), size=n_regular)
    y_regular = a * x_regular + b + noise_regular

    #High variance on x
    x_leverage = np.random.uniform(low=leverage_x_range[0], high=leverage_x_range[1], size=n_special) * np.random.choice([-1, 1], n_special)
    noise_leverage = np.random.normal(loc=0, scale=np.sqrt(variance), size=n_special)
    y_leverage = a * x_leverage + b + noise_leverage

    #High variance on y
    x_outlier = np.random.normal(loc=0, scale=regular_x_scale, size=n_special)
    y_outlier = a * x_outlier + b + np.random.uniform(low=outlier_y_error_range[0], high=outlier_y_error_range[1], size=n_special) * np.random.choice([-1, 1], n_special)

    #High variance on both x and y
    x_leverage_outlier = np.random.uniform(low=leverage_x_range[0], high=leverage_x_range[1], size=n_special) * np.random.choice([-1, 1], n_special)
    y_leverage_outlier = a * x_leverage_outlier + b + np.random.uniform(low=outlier_y_error_range[0], high=outlier_y_error_range[1], size=n_special) * np.random.choice([-1, 1], n_special)


    X_all = np.concatenate([x_regular, x_leverage, x_outlier, x_leverage_outlier])
    Y_all = np.concatenate([y_regular, y_leverage, y_outlier, y_leverage_outlier])

    design_matrix = np.vstack([X_all, np.ones(len(X_all))]).T
    hat_matrix = design_matrix @ np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T
    leverage_scores = np.diag(hat_matrix)
    highest_leverage_idx = np.argmax(leverage_scores)

    ax.set_title(f'σ² = {variance}')
    ax.grid(True, alpha=0.3)
    ax.scatter(x_regular, y_regular, c='blue', label='Regular', alpha=0.6)
    ax.scatter(x_outlier, y_outlier, c='green', marker='s', s=80, label='Outlier (Y)')
    ax.scatter(x_leverage, y_leverage, c='orange', marker='D', s=80, label='Leverage (X)')
    ax.scatter(x_leverage_outlier, y_leverage_outlier, c='purple', marker='^', s=80, label='Leverage Outlier (X,Y)')
    line_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(line_x, a * line_x + b, 'k--', label='True Model')
    ax.scatter(X_all[highest_leverage_idx], Y_all[highest_leverage_idx],
               facecolors='none', edgecolors='red', s=200, linewidth=2,
               label=f'Highest Leverage (Score: {leverage_scores[highest_leverage_idx]:.2f})')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print("\n--- Exercise 1.2: KNN on Clustered Data ---")

contamination = 0.1
X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    contamination=contamination,
    random_state=42
)

def plot_comparison(ax, X, labels, title):
    """Helper function to create a scatter plot with consistent coloring."""
    colors = np.array(['blue', 'red'])
    ax.scatter(X[:, 0], X[:, 1], c=colors[labels.astype(int)], alpha=0.7)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

n_neighbors_list = [5, 15, 50]

for n_neighbors in n_neighbors_list:
    print(f"\n--- Testing with n_neighbors = {n_neighbors} ---")
    clf = KNN(n_neighbors=n_neighbors, contamination=contamination)
    clf.fit(X_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'KNN Performance with n_neighbors = {n_neighbors}', fontsize=16)

    plot_comparison(axes[0, 0], X_train, y_train, 'Ground Truth (Train)')

    plot_comparison(axes[0, 1], X_train, y_train_pred, 'Predicted (Train)')

    plot_comparison(axes[1, 0], X_test, y_test, 'Ground Truth (Test)')
    
    plot_comparison(axes[1, 1], X_test, y_test_pred, 'Predicted (Test)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    balanced_acc = skl.balanced_accuracy_score(y_test, y_test_pred)
    print(f"Balanced Accuracy on Test Data: {balanced_acc:.4f}")


print("\n--- Exercise 1.3: KNN vs LOF on Different Density Clusters ---")


n_samples_list = [200, 100]
cluster_std_list = [2.0, 6.0]  # Standard deviations as specified in exercise
centers = [(-10, -10), (10, 10)]

X_cluster1, _ = make_blobs(n_samples=n_samples_list[0], 
                           n_features=2,
                           centers=[centers[0]], 
                           cluster_std=cluster_std_list[0], 
                           random_state=42)

X_cluster2, _ = make_blobs(n_samples=n_samples_list[1], 
                           n_features=2,
                           centers=[centers[1]], 
                           cluster_std=cluster_std_list[1], 
                           random_state=42)

# Combine the two inlier clusters
X_inliers = np.vstack([X_cluster1, X_cluster2])

# 2. Generate a separate set of outlier points
contamination_rate = 0.07
n_inliers = len(X_inliers)
n_outliers = int((n_inliers * contamination_rate) / (1 - contamination_rate))

# Generate outliers uniformly scattered across the space
np.random.seed(42)
X_outliers = np.random.uniform(low=-20, high=20, size=(n_outliers, 2))

# 3. Combine inliers and outliers
X_combined = np.vstack([X_inliers, X_outliers])
y_combined = np.array([0] * n_inliers + [1] * n_outliers)

print(f"Total samples: {len(X_combined)}")
print(f"Cluster 1 (dense): {n_samples_list[0]} samples, std={cluster_std_list[0]}")
print(f"Cluster 2 (sparse): {n_samples_list[1]} samples, std={cluster_std_list[1]}")
print(f"Number of outliers: {n_outliers}")
print(f"Actual contamination rate: {n_outliers / len(X_combined):.3f}")

n_neighbors_list = [5, 15, 50]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('KNN vs LOF on Clusters with Different Densities (std=2 vs std=6)', fontsize=16)

for col_idx, n_neighbors in enumerate(n_neighbors_list):
    print(f"\n--- Testing with n_neighbors = {n_neighbors} ---")
    

    knn_model = KNN(n_neighbors=n_neighbors, contamination=contamination_rate)
    knn_model.fit(X_combined)
    y_pred_knn = knn_model.predict(X_combined)
    

    balanced_acc_knn = skl.balanced_accuracy_score(y_combined, y_pred_knn)
    print(f"KNN Balanced Accuracy: {balanced_acc_knn:.4f}")
    
    ax_knn = axes[0, col_idx]
    colors = np.array(['blue', 'red'])
    ax_knn.scatter(X_combined[:, 0], X_combined[:, 1], 
                   c=colors[y_pred_knn.astype(int)], 
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=0.5)
    ax_knn.set_title(f'KNN (n_neighbors={n_neighbors})\nBalanced Acc: {balanced_acc_knn:.3f}')
    ax_knn.set_xlabel('Feature 1')
    ax_knn.grid(True, alpha=0.3)
    if col_idx == 0:
        ax_knn.set_ylabel('KNN Predictions\nFeature 2', fontsize=11, fontweight='bold')
    
    lof_model = LOF(n_neighbors=n_neighbors, contamination=contamination_rate)
    lof_model.fit(X_combined)
    y_pred_lof = lof_model.predict(X_combined)
    
    balanced_acc_lof = skl.balanced_accuracy_score(y_combined, y_pred_lof)
    print(f"LOF Balanced Accuracy: {balanced_acc_lof:.4f}")
    
    ax_lof = axes[1, col_idx]
    ax_lof.scatter(X_combined[:, 0], X_combined[:, 1], 
                   c=colors[y_pred_lof.astype(int)], 
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=0.5)
    ax_lof.set_title(f'LOF (n_neighbors={n_neighbors})\nBalanced Acc: {balanced_acc_lof:.3f}')
    ax_lof.set_xlabel('Feature 1')
    ax_lof.grid(True, alpha=0.3)
    if col_idx == 0:
        ax_lof.set_ylabel('LOF Predictions\nFeature 2', fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# --- Exercise 1.4: Ensemble Methods on Cardio Dataset ---

print("\n--- Exercise 1.4: Ensemble Methods on Cardio Dataset ---")

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization


print("Loading cardio dataset...")
mat_data = loadmat('cardio.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of outliers: {sum(y)}")
contamination_cardio = sum(y) / len(y)
print(f"Contamination rate: {contamination_cardio:.4f}")


X_train_raw, X_test_raw, y_train_cardio, y_test_cardio = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_norm, X_test_norm = standardizer(X_train_raw, X_test_raw)

print(f"\nTraining samples: {len(X_train_norm)}")
print(f"Test samples: {len(X_test_norm)}")

n_neighbors_range = range(30, 121, 10)
n_models = len(n_neighbors_range)

print(f"\nCreating ensemble of {n_models} models with n_neighbors: {list(n_neighbors_range)}")

train_scores_list = []
test_scores_list = []

print("\n--- Training KNN models ---")
for n_neighbors in n_neighbors_range:
    print(f"\nTraining KNN with n_neighbors = {n_neighbors}")

    model = KNN(n_neighbors=n_neighbors, contamination=contamination_cardio)
    model.fit(X_train_norm)

    train_scores = model.decision_function(X_train_norm)
    test_scores = model.decision_function(X_test_norm)

    train_scores_list.append(train_scores)
    test_scores_list.append(test_scores)

    y_train_pred = model.predict(X_train_norm)
    y_test_pred = model.predict(X_test_norm)

    ba_train = skl.balanced_accuracy_score(y_train_cardio, y_train_pred)
    ba_test = skl.balanced_accuracy_score(y_test_cardio, y_test_pred)
    
    print(f"  Train BA: {ba_train:.4f}")
    print(f"  Test BA: {ba_test:.4f}")

train_scores_matrix = np.column_stack(train_scores_list)
test_scores_matrix = np.column_stack(test_scores_list)

print(f"\nTrain scores matrix shape: {train_scores_matrix.shape}")
print(f"Test scores matrix shape: {test_scores_matrix.shape}")

print("\nNormalizing ensemble scores...")
train_scores_norm, test_scores_norm = standardizer(train_scores_matrix, test_scores_matrix)

print("\n--- Strategy 1: Average ---")
train_scores_avg = average(train_scores_norm)
test_scores_avg = average(test_scores_norm)

threshold_avg = np.quantile(train_scores_avg, 1 - contamination_cardio)
print(f"Threshold (average): {threshold_avg:.4f}")

y_train_pred_avg = (train_scores_avg > threshold_avg).astype(int)
y_test_pred_avg = (test_scores_avg > threshold_avg).astype(int)

ba_train_avg = skl.balanced_accuracy_score(y_train_cardio, y_train_pred_avg)
ba_test_avg = skl.balanced_accuracy_score(y_test_cardio, y_test_pred_avg)

print(f"Train BA (Average): {ba_train_avg:.4f}")
print(f"Test BA (Average): {ba_test_avg:.4f}")

print("\n--- Strategy 2: Maximization ---")
train_scores_max = maximization(train_scores_norm)
test_scores_max = maximization(test_scores_norm)

threshold_max = np.quantile(train_scores_max, 1 - contamination_cardio)
print(f"Threshold (maximization): {threshold_max:.4f}")

y_train_pred_max = (train_scores_max > threshold_max).astype(int)
y_test_pred_max = (test_scores_max > threshold_max).astype(int)

ba_train_max = skl.balanced_accuracy_score(y_train_cardio, y_train_pred_max)
ba_test_max = skl.balanced_accuracy_score(y_test_cardio, y_test_pred_max)

print(f"Train BA (Maximization): {ba_train_max:.4f}")
print(f"Test BA (Maximization): {ba_test_max:.4f}")

print("\n--- Summary ---")
print(f"Average Strategy    - Train BA: {ba_train_avg:.4f}, Test BA: {ba_test_avg:.4f}")
print(f"Maximization Strategy - Train BA: {ba_train_max:.4f}, Test BA: {ba_test_max:.4f}")

