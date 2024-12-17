import numpy as np

# -------------------------
# Load Data
# -------------------------
print("Loading train and test embeddings...")
train_data = np.load("train_embeddings.npz")
test_data = np.load("test_embeddings.npz")

# Extract embeddings and labels
train_seq1 = train_data['seq1']
train_seq2 = train_data['seq2']
y_train = train_data['labels']

test_seq1 = test_data['seq1']
test_seq2 = test_data['seq2']
y_test = test_data['labels']

# Concatenate train embeddings for normalization
train_combined = np.concatenate((train_seq1, train_seq2), axis=1)  # Shape: (samples, 1536)
test_combined = np.concatenate((test_seq1, test_seq2), axis=1)  # Shape: (samples, 1536)

# -------------------------
# Compute Min-Max Parameters
# -------------------------
global_min = np.min(train_combined)  # Min of all training data
global_max = np.max(train_combined)  # Max of all training data

# -------------------------
# Apply Min-Max Normalization to Range [-1, 1]
# -------------------------
train_combined_normalized = 2 * (train_combined - global_min) / (global_max - global_min) - 1
test_combined_normalized = 2 * (test_combined - global_min) / (global_max - global_min) - 1  # Use train min/max

# Split normalized data back into seq1 and seq2
train_seq1_normalized = train_combined_normalized[:, :train_seq1.shape[1]]
train_seq2_normalized = train_combined_normalized[:, train_seq1.shape[1]:]
test_seq1_normalized = test_combined_normalized[:, :test_seq1.shape[1]]
test_seq2_normalized = test_combined_normalized[:, test_seq1.shape[1]:]

# -------------------------
# Save Normalized Data
# -------------------------
print("Saving normalized train and test embeddings...")
np.savez("train_normalized_embeddings_minmax.npz", seq1=train_seq1_normalized, seq2=train_seq2_normalized, labels=y_train)
np.savez("test_normalized_embeddings_minmax.npz", seq1=test_seq1_normalized, seq2=test_seq2_normalized, labels=y_test)

print("Normalization to range [-1, 1] complete. Files saved as 'train_normalized_embeddings_minmax.npz' and 'test_normalized_embeddings_minmax.npz'")
