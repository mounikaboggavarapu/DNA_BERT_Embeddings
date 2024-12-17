import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
from scipy.stats import skew, kurtosis

# Load the .npz file
data = np.load("train_embeddings.npz")

# Access seq1 and seq2 embeddings
seq1_embeddings = torch.tensor(data['seq1']).cuda()
seq2_embeddings = torch.tensor(data['seq2']).cuda()

# Concatenate seq1 and seq2 along the last axis
combined_embeddings = torch.cat((seq1_embeddings, seq2_embeddings), dim=1)  # Shape: (samples, 1536)

# Flatten combined embeddings for analysis
flattened_embeddings = combined_embeddings.view(-1).cpu().numpy()

# Compute statistical metrics
mean_val = flattened_embeddings.mean()
std_dev = flattened_embeddings.std()
min_val = flattened_embeddings.min()
max_val = flattened_embeddings.max()
skewness = skew(flattened_embeddings)
kurt = kurtosis(flattened_embeddings)

# Print metrics
print("Statistical Analysis of Combined Embeddings:")
print(f"Mean: {mean_val:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Min: {min_val:.4f}")
print(f"Max: {max_val:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# -------------------------------
# Plot histogram with custom bins
# -------------------------------
plt.figure(figsize=(16, 8))
bins = np.arange(min_val, max_val + 0.1, 0.1)  # Define bins with step 0.1
plt.hist(flattened_embeddings, bins=bins, alpha=0.7, edgecolor='black')
plt.title("Histogram of Combined Seq1 and Seq2 Embeddings", fontsize=16)
plt.xlabel("Embedding Values (1 Decimal Place)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

# Format x-axis to show 1 decimal place
formatter = FuncFormatter(lambda x, _: f'{x:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)

# Disable scientific notation on y-axis
plt.ticklabel_format(axis='y', style='plain')

# Add gridlines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the histogram
plt.tight_layout()
plt.savefig("histogram2.png")
plt.close()
print("Histogram saved as combined_embeddings_histogram_custom.png")

# -------------------------
# Plot boxplot of embeddings
# -------------------------
plt.figure(figsize=(12, 6))
plt.boxplot(flattened_embeddings, vert=False, patch_artist=True, showmeans=True, boxprops=dict(facecolor='lightblue'))
plt.title("Boxplot of Combined Seq1 and Seq2 Embeddings", fontsize=16)
plt.xlabel("Embedding Values", fontsize=14)

# Format x-axis for 1 decimal place
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

# Save the boxplot
plt.tight_layout()
plt.savefig("boxplot2.png")
plt.close()
print("Boxplot saved as combined_embeddings_boxplot.png")
