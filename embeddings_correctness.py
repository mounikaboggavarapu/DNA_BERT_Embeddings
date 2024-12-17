import numpy as np

# Load the .npz file
data = np.load("train_embeddings.npz")

# List all the arrays stored in the .npz file
print("Contents of the .npz file:", data.files)

# Access and inspect arrays
seq1_embeddings = data['seq1']
seq2_embeddings = data['seq2']
labels = data['labels']

print("Seq1 Embedding Shape:", seq1_embeddings.shape)
print("Seq2 Embedding Shape:", seq2_embeddings.shape)
print("Labels Shape:", labels.shape)

# Example: Inspect first embedding
print("First Seq1 Embedding:", seq1_embeddings[0])
print("First Seq2 Embedding:", seq2_embeddings[0])
print("First Label:", labels[0])

# Close the file to release memory
data.close()
