import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# 1. Load Dataset
# ==============================
# File path for the training data
train_file_path = "train_normalized_embeddings_minmax_10percent.npz"

# Load normalized training data
print("Loading training data...")
try:
    train_data = np.load(train_file_path)
    train_seq1 = train_data['seq1']  # shape: (num_samples, 768)
    train_seq2 = train_data['seq2']  # shape: (num_samples, 768)
    y_train = train_data['labels'].astype(int)  # shape: (num_samples,)
    print(f"Data loaded successfully: seq1 shape = {train_seq1.shape}, seq2 shape = {train_seq2.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# ==============================
# 2. Compute Interaction Features
# ==============================
print("Computing interaction features...")
difference = train_seq1 - train_seq2
product = train_seq1 * train_seq2

# Combine features into one array
combined_train = np.concatenate((train_seq1, train_seq2, difference, product), axis=1)
print(f"Combined feature shape: {combined_train.shape}")

# ==============================
# 3. Apply t-SNE
# ==============================
print("Applying t-SNE for 3D embedding on the entire dataset...")
tsne = TSNE(n_components=3, random_state=42, perplexity=30)

# Compute t-SNE embeddings for the full dataset
tsne_results_3d = tsne.fit_transform(combined_train)
print("t-SNE computation complete.")

# ==============================
# 4. Plot t-SNE in 3D
# ==============================
print("Creating 3D plot...")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D space
scatter = ax.scatter(
    tsne_results_3d[:, 0], tsne_results_3d[:, 1], tsne_results_3d[:, 2],
    c=y_train, cmap='viridis', s=10, alpha=0.8
)

# Add titles and labels
ax.set_title('t-SNE 3D Visualization of Full Dataset', fontsize=16)
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')

# Add a color bar
colorbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.5)
colorbar.set_label('Class Labels')

# Save the plot
plt.savefig("tsne_3d_full_dataset.png", dpi=300)
plt.show()
