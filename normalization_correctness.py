import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# -------------------------
# Load Data
# -------------------------
print("Loading normalized embeddings...")
data = np.load("train_normalized_embeddings_minmax.npz")

# Access seq1 and seq2 embeddings
seq1_embeddings = data['seq1']
seq2_embeddings = data['seq2']

# Concatenate seq1 and seq2
combined_embeddings = np.concatenate((seq1_embeddings, seq2_embeddings), axis=1)
combined_flattened = combined_embeddings.flatten()

# -------------------------
# Compute Statistical Metrics
# -------------------------
mean = np.mean(combined_flattened)
std_dev = np.std(combined_flattened)
min_val = np.min(combined_flattened)
max_val = np.max(combined_flattened)
data_skewness = skew(combined_flattened)
data_kurtosis = kurtosis(combined_flattened)

# Print metrics
print("Statistical Analysis of Combined Embeddings:")
print(f"Mean: {mean:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Min: {min_val:.4f}")
print(f"Max: {max_val:.4f}")
print(f"Skewness: {data_skewness:.4f}")
print(f"Kurtosis: {data_kurtosis:.4f}")

# -------------------------
# Histogram of Combined Embeddings
# -------------------------
plt.figure(figsize=(16, 8))
plt.hist(combined_flattened, bins=50, alpha=0.7, edgecolor='black')
plt.title("Histogram of Combined Embeddings (Seq1 + Seq2)", fontsize=16)
plt.xlabel("Embedding Values", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("combined_embeddings_histogram.png")
plt.close()
print("Histogram saved as combined_embeddings_histogram.png")

# -------------------------
# Boxplot for Combined Embeddings (Downsampled)
# -------------------------
downsampled_size = 10000  # Adjust this value based on available memory
if len(combined_flattened) > downsampled_size:
    combined_flattened_downsampled = np.random.choice(combined_flattened, downsampled_size, replace=False)
else:
    combined_flattened_downsampled = combined_flattened

plt.figure(figsize=(12, 6))
plt.boxplot(combined_flattened_downsampled, vert=False, patch_artist=True, showmeans=True)
plt.title("Boxplot of Combined Embeddings (Downsampled)", fontsize=16)
plt.xlabel("Embedding Values", fontsize=14)
plt.tight_layout()
plt.savefig("combined_embeddings_boxplot_downsampled.png")
plt.close()
print("Boxplot saved as combined_embeddings_boxplot_downsampled.png")
