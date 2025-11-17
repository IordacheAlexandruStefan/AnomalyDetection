import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from pyod.utils.utility import standardizer

np.random.seed(42)

# Generate 500 2D points from a standard normal distribution
X_train, _ = make_blobs(n_samples=500,
                        centers=[(0, 0)],
                        n_features=2,
                        cluster_std=1.0,
                        random_state=42)

print(f"Shape of training data: {X_train.shape}")

n_projections = 5
n_bins = 20 # We'll start with 20 bins, as suggested 

# 1. Generate 5 random unit-length projection vectors 
mean = [0, 0]
cov = [[1, 0], [0, 1]] # Identity matrix
projections = np.random.multivariate_normal(mean, cov, n_projections)
# Normalize to unit length
projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

# 2. Project the training data
projected_data = X_train @ projections.T # Shape (500, 5)

# 3. Compute histograms and probabilities [cite: 56, 57]
histograms = []
bin_edges_list = []

for i in range(n_projections):
    data_min = projected_data[:, i].min()
    data_max = projected_data[:, i].max()
    data_range = data_max - data_min
    # Use a more generous padding (e.g., 20% on each side)
    padding = data_range * 0.2
    hist_range = (data_min - padding, data_max + padding)
    
    hist, bin_edges = np.histogram(projected_data[:, i],
                                 bins=n_bins,
                                 range=hist_range,
                                 density=False)
    
    probs = hist / X_train.shape[0]
    histograms.append(probs)
    bin_edges_list.append(bin_edges)

    # 1. Generate test data
X_test = np.random.uniform(low=-3.0, high=3.0, size=(500, 2))

# 2. Function to calculate scores
def calculate_anomaly_scores(X_points, projections, histograms, bin_edges_list):
    all_scores = []
    
    for x in X_points: # For each test point
        projected_point = x @ projections.T # Project it
        point_probs = []
        
        for i in range(projections.shape[0]):

            proj_val = projected_point[i]
            edges = bin_edges_list[i]

            bin_index = np.searchsorted(edges, proj_val, side='right') - 1
            
            # Check if it's outside the histogram range
            if bin_index < 0 or bin_index >= len(histograms[i]):
                point_probs.append(0.0) # 0 probability = anomaly
            else:
                point_probs.append(histograms[i][bin_index])
        
        # Score = mean of probabilities 
        all_scores.append(np.mean(point_probs))
        
    return np.array(all_scores)

scores = calculate_anomaly_scores(X_test, projections, histograms, bin_edges_list)

plt.figure(figsize=(10, 8))

plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap='viridis')
plt.colorbar(label='Mean Probability (Anomaly Score)')
plt.title(f'Simple LODA Score Map (n_bins = {n_bins})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# --- Testing Different Number of Bins ---
print("\n--- Testing Different Number of Bins ---")

bin_values = [10, 20, 50, 100]  # Different bin counts to test

# First, calculate all scores to determine the global min/max for consistent coloring
all_scores_dict = {}
vmin = float('inf')
vmax = float('-inf')

for n_bins_test in bin_values:
    histograms_test = []
    bin_edges_list_test = []
    
    for i in range(n_projections):
        data_min = projected_data[:, i].min()
        data_max = projected_data[:, i].max()
        data_range = data_max - data_min
        padding = data_range * 0.2
        hist_range = (data_min - padding, data_max + padding)
        
        hist, bin_edges = np.histogram(projected_data[:, i],
                                     bins=n_bins_test,
                                     range=hist_range,
                                     density=False)
        
        probs = hist / X_train.shape[0]
        histograms_test.append(probs)
        bin_edges_list_test.append(bin_edges)
    
    scores_test = calculate_anomaly_scores(X_test, projections, 
                                          histograms_test, bin_edges_list_test)
    all_scores_dict[n_bins_test] = scores_test
    
    # Update global min/max
    vmin = min(vmin, scores_test.min())
    vmax = max(vmax, scores_test.max())

print(f"\nGlobal score range: [{vmin:.4f}, {vmax:.4f}]")

# Now plot with consistent color scale
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('LODA Score Maps with Different Number of Bins', fontsize=16)

for idx, n_bins_test in enumerate(bin_values):
    print(f"\nPlotting n_bins = {n_bins_test}")
    scores_test = all_scores_dict[n_bins_test]
    
    # Plot with fixed vmin and vmax
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], 
                        c=scores_test, cmap='viridis', alpha=0.6,
                        vmin=vmin, vmax=vmax)  # Fixed scale
    ax.set_title(f'n_bins = {n_bins_test}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.colorbar(scatter, ax=ax, label='Anomaly Score (Mean Probability)')
    
    # Print some statistics
    print(f"  Score range: [{scores_test.min():.4f}, {scores_test.max():.4f}]")
    print(f"  Mean score: {scores_test.mean():.4f}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()