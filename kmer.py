import torch
import numpy as np
from tqdm import tqdm
import warnings
import gc
# Removed: import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# 1. Preprocessing Functions
# ============================================================

def debug_print(message, level="INFO"):
    print(f"[{level}] {message}")

def generate_ngrams(sequence, n=3):
    """Generate n-grams from a sequence."""
    return [sequence[i:i+n] for i in range(len(sequence) - n + 1)]

def create_vocab(sequences, n=3):
    """Create a vocabulary of n-grams from a list of sequences."""
    vocab = {}
    debug_print(f"Creating vocabulary with {n}-grams...")
    for seq in tqdm(sequences, desc="Building Vocabulary"):
        for gram in generate_ngrams(seq, n):
            if gram not in vocab:
                vocab[gram] = len(vocab)  # Assign a unique integer ID
    debug_print(f"Vocabulary size: {len(vocab)}")
    return vocab

def embed_ngrams(ngrams, vocab, embed_dim, device):
    """Embed n-grams using GPU-based one-hot encoding."""
    embedding = torch.zeros((len(ngrams), embed_dim), device=device)
    for i, gram in enumerate(ngrams):
        if gram in vocab:
            embedding[i, vocab[gram] % embed_dim] = 1
    return embedding.flatten()

def min_max_normalize_tensor(tensor, feature_range=(-1, 1)):
    """Apply Min-Max Normalization to a GPU tensor."""
    min_val = tensor.min()
    max_val = tensor.max()
    range_min, range_max = feature_range
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor * (range_max - range_min) + range_min

def load_data(file_path):
    """Load data from a text file."""
    debug_print(f"Loading data from {file_path}...")
    data = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            seq1, seq2, label = line.strip().split("\t")
            data.append((seq1, seq2, label))
    debug_print(f"Total rows loaded: {len(data)} from {file_path}")
    return data

# ============================================================
# 2. Main Preprocessing Routine (In-Memory)
# ============================================================

def preprocess_data(train_file, test_file, embed_dim=128, n=3, feature_range=(-1,1), device=None):
    # Load raw data
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    # *** Commented out the 1% usage line ***
    # train_size = int(len(train_data) * 0.01)
    # train_data = train_data[:train_size]
    debug_print(f"Using all {len(train_data)} training samples (100% of original)")
    debug_print(f"Using all {len(test_data)} test samples")

    # Build vocabulary from the training data only
    all_sequences = [row[0] for row in train_data] + [row[1] for row in train_data]
    vocab = create_vocab(all_sequences, n=n)

    def preprocess_split(data_split):
        seq1_embeds, seq2_embeds, labels = [], [], []
        for idx, row in enumerate(tqdm(data_split, desc="Processing Split")):
            seq1_grams = generate_ngrams(row[0], n=n)
            seq2_grams = generate_ngrams(row[1], n=n)

            seq1_embed = embed_ngrams(seq1_grams, vocab, embed_dim, device)
            seq2_embed = embed_ngrams(seq2_grams, vocab, embed_dim, device)

            # Normalize
            seq1_norm = min_max_normalize_tensor(seq1_embed, feature_range=feature_range).cpu().numpy()
            seq2_norm = min_max_normalize_tensor(seq2_embed, feature_range=feature_range).cpu().numpy()

            labels.append(int(row[2] == "TRUE"))
            seq1_embeds.append(seq1_norm)
            seq2_embeds.append(seq2_norm)

        return np.array(seq1_embeds), np.array(seq2_embeds), np.array(labels)

    train_seq1, train_seq2, train_labels = preprocess_split(train_data)
    test_seq1, test_seq2, test_labels = preprocess_split(test_data)

    return train_seq1, train_seq2, train_labels, test_seq1, test_seq2, test_labels

# ============================================================
# 3. Custom Dataset Class (From Memory)
# ============================================================

class InteractionDataset(Dataset):
    def __init__(self, seq1, seq2, labels):
        self.seq1 = seq1
        self.seq2 = seq2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq1 = torch.tensor(self.seq1[idx], dtype=torch.float32)
        seq2 = torch.tensor(self.seq2[idx], dtype=torch.float32)

        difference = seq1 - seq2
        product = seq1 * seq2
        combined = torch.cat((seq1, seq2, difference, product), dim=0)  # shape: (3072,)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return combined, label

# ============================================================
# 4. Model Architecture
# ============================================================

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
        
        self.fc_out = nn.Linear(hidden_sizes[2], 2)
        
        self.residual1 = nn.Linear(input_size, hidden_sizes[1])
        self.residual2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        
    def forward(self, x):
        out1 = self.fc1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.dropout1(out1)
        
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out2 = self.dropout2(out2)
        res2 = self.residual1(x)
        out2 = out2 + res2
        
        out3 = self.fc3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        out3 = self.dropout3(out3)
        res3 = self.residual2(out2)
        out3 = out3 + res3
        
        output = self.fc_out(out3)
        return output

# ============================================================
# 5. Early Stopping Implementation
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
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

# ============================================================
# 6. Training and Evaluation Functions
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, early_stopper, device):
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", leave=False)
        for X_batch, y_batch in train_progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct_preds += (preds == y_batch).sum().item()
            total_preds += y_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_preds / total_preds
        print(f"Epoch [{epoch}/{epochs}] Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
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
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

def evaluate_model(model, loader, threshold, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)[:,1]
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

def find_best_threshold(loader, model, device):
    model.eval()
    thresholds = np.linspace(0.1, 0.9, 17)
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

# ============================================================
# 7. Main Execution
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocess data (no saving, just in-memory)
    train_seq1, train_seq2, train_labels, test_seq1, test_seq2, test_labels = preprocess_data(
        train_file="genomic-2h-train.txt",
        test_file="genomic-2h-test.txt",
        embed_dim=128,
        n=3,
        feature_range=(-1,1),
        device=device
    )

    full_train_dataset = InteractionDataset(train_seq1, train_seq2, train_labels)
    test_dataset = InteractionDataset(test_seq1, test_seq2, test_labels)

    print("Splitting into training and validation sets...")
    train_indices, val_indices = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=full_train_dataset.labels
    )

    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_train_dataset, val_indices)

    print("Computing class weights for training subset...")
    train_labels_sub = [full_train_dataset.labels[idx] for idx in train_indices]
    class_counts = np.bincount(train_labels_sub)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    print("Creating WeightedRandomSampler...")
    train_weights = [class_weights[label] for label in train_labels_sub]
    train_weights = torch.tensor(train_weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    batch_size = 64
    num_workers = 4

    print("Creating DataLoaders...")
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    del full_train_dataset, train_subset, val_subset
    gc.collect()

    input_size = 3072
    hidden_sizes = [1024, 512, 256]
    dropout_rate = 0.3

    print("\nInitializing the model...")
    model = ResidualFCNN(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    early_stopper = EarlyStopping(patience=7, min_delta=0.0)

    epochs = 50
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, early_stopper, device)

    print("\nFinding the best threshold based on validation F1 score...")
    best_threshold = find_best_threshold(val_loader, model, device)
    print(f"Best Threshold: {best_threshold:.2f}")

    print("\nEvaluating the model on the Test Set...")
    test_metrics = evaluate_model(model, test_loader, best_threshold, device)
    print("\nTest Set Evaluation Metrics:")
    print(f"Accuracy: {test_metrics[0]:.4f}")
    print(f"Precision: {test_metrics[1]:.4f}")
    print(f"Recall: {test_metrics[2]:.4f}")
    print(f"F1 Score: {test_metrics[3]:.4f}")
    print(f"ROC-AUC: {test_metrics[5]:.4f}")
    print(f"PR-AUC: {test_metrics[6]:.4f}")
    print("Confusion Matrix:")
    print(test_metrics[4])

    model_save_path = "residual_fcnn_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")
    print("\nTraining and evaluation complete.")
