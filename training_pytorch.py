import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import gc  # Garbage collector

# ==============================
# 1. Environment Setup
# ==============================

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# 2. Custom Dataset Class
# ==============================

class InteractionDataset(Dataset):
    def __init__(self, npz_path):
        """
        Custom Dataset to handle large datasets using memory mapping.

        Args:
            npz_path (str): Path to the .npz file containing 'seq1', 'seq2', and 'labels'.
        """
        self.data = np.load(npz_path, mmap_mode='r')  # Memory-mapped access
        self.seq1 = self.data['seq1']  # shape: (N, 768)
        self.seq2 = self.data['seq2']  # shape: (N, 768)
        self.labels = self.data['labels'].astype(int)  # shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve sequences
        seq1 = self.seq1[idx]  # shape: (768,)
        seq2 = self.seq2[idx]  # shape: (768,)

        # Convert to torch tensors
        seq1 = torch.tensor(seq1, dtype=torch.float32)
        seq2 = torch.tensor(seq2, dtype=torch.float32)

        # Compute interaction features
        difference = seq1 - seq2  # Element-wise difference
        product = seq1 * seq2     # Element-wise product

        # Concatenate original and interaction features
        combined = torch.cat((seq1, seq2, difference, product), dim=0)  # shape: (3072,)

        # Retrieve label
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return combined, label

# ==============================
# 3. Data Loading and Preprocessing
# ==============================

# File paths for train and test data
train_file_path = "train_normalized_embeddings_minmax.npz"
test_file_path = "test_normalized_embeddings_minmax.npz"

# Initialize Datasets
print("Initializing Datasets...")
full_train_dataset = InteractionDataset(train_file_path)
test_dataset = InteractionDataset(test_file_path)

# Split full_train_dataset into training and validation sets
print("Splitting into training and validation sets...")
train_indices, val_indices = train_test_split(
    np.arange(len(full_train_dataset)),
    test_size=0.2,
    random_state=42,
    stratify=full_train_dataset.labels
)

# Create subset datasets
train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
val_subset = torch.utils.data.Subset(full_train_dataset, val_indices)

# Compute class weights for handling imbalance based on training subset
print("Computing class weights for training subset...")
train_labels = [full_train_dataset.labels[idx] for idx in train_indices]
class_counts = np.bincount(train_labels)
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

# Create WeightedRandomSampler for balanced training
print("Creating WeightedRandomSampler...")
train_weights = [class_weights[label] for label in train_labels]
train_weights = torch.tensor(train_weights, dtype=torch.float32)
sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

# Create DataLoaders
batch_size = 64  # Adjust based on your GPU memory
num_workers = 4  # Adjust based on your CPU cores

print("Creating DataLoaders...")
train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print("\nDataLoaders created successfully.")
print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Test samples: {len(test_dataset)}")

# Clear variables to free memory
del full_train_dataset
del train_subset
del val_subset
gc.collect()

# ==============================
# 4. Model Architecture
# ==============================

class ResidualFCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(ResidualFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc_out = nn.Linear(hidden_sizes[2], 2)  # Outputting 2 logits for binary classification
        
        # Residual connections
        self.residual1 = nn.Linear(input_size, hidden_sizes[1])
        self.residual2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        
    def forward(self, x):
        # First layer
        out1 = self.fc1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.dropout1(out1)
        
        # Second layer with residual
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out2 = self.dropout2(out2)
        res2 = self.residual1(x)
        out2 = out2 + res2  # Residual connection
        
        # Third layer with residual
        out3 = self.fc3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        out3 = self.dropout3(out3)
        res3 = self.residual2(out2)
        out3 = out3 + res3  # Residual connection
        
        # Output layer
        output = self.fc_out(out3)
        return output  # Returns logits for both classes

# Instantiate the ResidualFCNN model
input_size = 3072  # 768 * 4 (seq1, seq2, difference, product)
hidden_sizes = [1024, 512, 256]
dropout_rate = 0.5

print("\nInitializing the model...")
model = ResidualFCNN(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
print(model)

# ==============================
# 5. Loss Function and Optimizer
# ==============================

criterion = nn.CrossEntropyLoss().to(device)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ==============================
# 6. Early Stopping Implementation
# ==============================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            print(f"Initial validation loss set to {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss improved to {self.best_loss:.4f}. Resetting counter.")

# ==============================
# 7. Training Function
# ==============================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, early_stopper):
    """
    Trains the model and evaluates on the validation set.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epochs (int): Number of epochs to train.
        early_stopper (EarlyStopping): Early stopping mechanism.
    """
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", leave=False)
        for X_batch, y_batch in train_progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct_preds += (preds == y_batch).sum().item()
            total_preds += y_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_preds / total_preds
        print(f"Epoch [{epoch}/{epochs}] Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val_preds = 0
        total_val_preds = 0
        
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch)
                val_running_loss += loss.item() * X_val_batch.size(0)
                
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                correct_val_preds += (preds == y_val_batch).sum().item()
                total_val_preds += y_val_batch.size(0)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_val_preds / total_val_preds
        print(f"Epoch [{epoch}/{epochs}] Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Step the scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        # Early stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

# ==============================
# 8. Evaluation Function
# ==============================

def evaluate_model(model, loader, threshold=0.5):
    """
    Evaluates the model on the provided DataLoader.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for evaluation data.
        threshold (float): Threshold for classification.

    Returns:
        tuple: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC, PR-AUC
    """
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)[:,1]  # Probability of the positive class
            preds = (probs >= threshold).long()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    return accuracy, precision, recall, f1, conf_matrix, roc_auc, pr_auc

# ==============================
# 9. Finding the Best Threshold
# ==============================

def find_best_threshold(loader, model):
    """
    Finds the best threshold to maximize the F1 score on the validation set.

    Args:
        loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The trained model.

    Returns:
        float: Best threshold value.
    """
    model.eval()
    thresholds = np.linspace(0.1, 0.9, 17)  # Steps of 0.05
    best_f1 = 0
    best_threshold = 0.5
    y_true, y_scores = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)[:,1].cpu().numpy()
            y_true.extend(y_batch.cpu().numpy())
            y_scores.extend(probs)
    
    for threshold in thresholds:
        preds = (np.array(y_scores) >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

# ==============================
# 10. Training the Model
# ==============================

# Initialize Early Stopping
early_stopper = EarlyStopping(patience=7, min_delta=0.0)

# Define number of epochs
epochs = 50  # You can adjust the number of epochs as needed

# Train the model
print("\nStarting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, early_stopper)

# ==============================
# 11. Finding the Best Threshold
# ==============================

# Find the best threshold based on validation set
print("\nFinding the best threshold based on validation F1 score...")
best_threshold = find_best_threshold(val_loader, model)
print(f"Best Threshold: {best_threshold:.2f}")

# ==============================
# 12. Evaluating the Model on Test Set
# ==============================

# Evaluate on Test Set
print("\nEvaluating the model on the Test Set...")
test_metrics = evaluate_model(model, test_loader, best_threshold)
print("\nTest Set Evaluation Metrics:")
print(f"Accuracy: {test_metrics[0]:.4f}")
print(f"Precision: {test_metrics[1]:.4f}")
print(f"Recall: {test_metrics[2]:.4f}")
print(f"F1 Score: {test_metrics[3]:.4f}")
print(f"ROC-AUC: {test_metrics[5]:.4f}")
print(f"PR-AUC: {test_metrics[6]:.4f}")
print("Confusion Matrix:")
print(test_metrics[4])

# ==============================
# 13. Saving the Model
# ==============================

model_save_path = "residual_fcnn_final.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}")

# ==============================
# 14. Summary
# ==============================

print("\nTraining and evaluation complete. Models have been saved.")
