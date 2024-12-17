import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import gc
import warnings

# ==============================
# 1. Environment Setup
# ==============================

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure GPU memory growth if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0].name}")
    except:
        print("Failed to set memory growth for GPU.")
else:
    print("No GPU available, using CPU")

# ==============================
# 2. Data Loading and Preprocessing
# ==============================

train_file_path = "train_normalized_embeddings_minmax.npz"
test_file_path = "test_normalized_embeddings_minmax.npz"

print("Loading normalized train and test embeddings...")
train_data = np.load(train_file_path)
test_data = np.load(test_file_path)

# Extract sequences and labels
train_seq1 = train_data['seq1']
train_seq2 = train_data['seq2']
y_train = train_data['labels'].astype(int)

test_seq1 = test_data['seq1']
test_seq2 = test_data['seq2']
y_test = test_data['labels'].astype(int)

print(f"Train seq1: {train_seq1.shape}, seq2: {train_seq2.shape}, labels: {y_train.shape}")
print(f"Test seq1: {test_seq1.shape}, seq2: {test_seq2.shape}, labels: {y_test.shape}")

# ==============================
# 3. Feature Engineering
# ==============================

print("Computing interaction features...")
# Compute difference and product features
difference_train = train_seq1 - train_seq2
product_train = train_seq1 * train_seq2

difference_test = test_seq1 - test_seq2
product_test = test_seq1 * test_seq2

# Concatenate all features
X_train_combined = np.concatenate((train_seq1, train_seq2, difference_train, product_train), axis=1)
X_test_combined = np.concatenate((test_seq1, test_seq2, difference_test, product_test), axis=1)

# Clean up unused variables to save memory
del train_seq1, train_seq2, difference_train, product_train
del test_seq1, test_seq2, difference_test, product_test
gc.collect()

# ==============================
# 4. Train-Validation Split
# ==============================

# Single-step split to ensure consistency across features and labels
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_combined,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test_combined.shape}")

# Clean up unused variables
del X_train_combined
gc.collect()

# Define inputs and outputs
# Here, X_train and X_val are the feature matrices, y_train_split and y_val are the labels
# X_test_combined is the test feature matrix, y_test is the test labels

# ==============================
# 5. Compute Class Weights
# ==============================

print("Computing class weights...")
classes = np.unique(y_train_split)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_split)
class_weights_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weights_dict}")

# ==============================
# 6. Model Architecture
# ==============================

def build_residual_fcnn(input_dim, hidden_sizes, dropout_rates, l2_regs):
    """
    Builds a Residual Fully Connected Neural Network (FCNN).

    Parameters:
    - input_dim: int, dimensionality of the input features.
    - hidden_sizes: list of ints, sizes of the hidden layers.
    - dropout_rates: list of floats, dropout rates for each hidden layer.
    - l2_regs: list of floats, L2 regularization factors for each hidden layer.

    Returns:
    - model: tf.keras.Model, the compiled model.
    """
    inputs = layers.Input(shape=(input_dim,), name='input_layer')

    # First Dense Layer
    x = layers.Dense(hidden_sizes[0], kernel_regularizer=regularizers.l2(l2_regs[0]), activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rates[0])(x)

    # First Residual Block
    residual = layers.Dense(hidden_sizes[1], kernel_regularizer=regularizers.l2(l2_regs[1]))(x)
    residual = layers.BatchNormalization()(residual)

    x_res = layers.Dense(hidden_sizes[1], activation='relu', kernel_regularizer=regularizers.l2(l2_regs[1]))(x)
    x_res = layers.BatchNormalization()(x_res)
    x_res = layers.Dropout(dropout_rates[1])(x_res)

    x = layers.Add()([x_res, residual])

    # Second Residual Block
    residual = layers.Dense(hidden_sizes[2], kernel_regularizer=regularizers.l2(l2_regs[2]))(x)
    residual = layers.BatchNormalization()(residual)

    x_res = layers.Dense(hidden_sizes[2], activation='relu', kernel_regularizer=regularizers.l2(l2_regs[2]))(x)
    x_res = layers.BatchNormalization()(x_res)
    x_res = layers.Dropout(dropout_rates[2])(x_res)

    x = layers.Add()([x_res, residual])

    # Output Layer
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Define hyperparameters based on the provided best values
input_dim = X_train.shape[1]
hidden_sizes = [768, 1024, 1024]        # units_1=768, units_2=1024, units_3=1024
dropout_rates = [0.4, 0.4, 0.4]         # dropout_1=0.4, dropout_2=0.4, dropout_3=0.4
l2_regs = [1e-06, 1e-05, 1e-05]         # l2_1=1e-06, l2_2=1e-05, l2_3=1e-05

print("Building model...")
model = build_residual_fcnn(input_dim, hidden_sizes, dropout_rates, l2_regs)
model.summary()

# ==============================
# 7. Training Setup
# ==============================

# Define optimizer with the specified learning rate
optimizer = optimizers.Adam(learning_rate=1e-3)  # learning_rate=0.001

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )
]

batch_size = 128
epochs = 50

print("Training model...")
history = model.fit(
    X_train,
    y_train_split,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights_dict,
    callbacks=callbacks_list,
    verbose=1
)

# ==============================
# 8. Best Threshold Determination
# ==============================

print("Determining the best threshold on validation data...")
# Predict probabilities on validation set
val_probs = model.predict(X_val, batch_size=batch_size)[:, 1]

# Function to find the best threshold based on F1 score
def find_best_threshold(y_true, y_probs):
    """
    Finds the threshold that maximizes the F1 score.

    Parameters:
    - y_true: array-like, true binary labels.
    - y_probs: array-like, predicted probabilities for the positive class.

    Returns:
    - best_threshold: float, the threshold with the highest F1 score.
    - best_f1: float, the highest F1 score achieved.
    """
    best_threshold = 0.5
    best_f1 = 0
    thresholds = np.arange(0.1, 0.9, 0.01)
    for threshold in thresholds:
        preds = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

best_threshold, best_f1 = find_best_threshold(y_val, val_probs)
print(f"Best threshold on validation data: {best_threshold:.2f} with F1 score: {best_f1:.4f}")

# ==============================
# 9. Evaluation
# ==============================

def evaluate_metrics(y_true, y_probs, threshold=0.5):
    """
    Evaluates various classification metrics.

    Parameters:
    - y_true: array-like, true binary labels.
    - y_probs: array-like, predicted probabilities for the positive class.
    - threshold: float, classification threshold.

    Returns:
    - metrics: dict, containing various evaluation metrics.
    """
    y_pred = (y_probs >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'pr_auc': average_precision_score(y_true, y_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

print("Evaluating on Training Data...")
train_probs = model.predict(X_train, batch_size=batch_size)[:, 1]
train_metrics = evaluate_metrics(y_train_split, train_probs, threshold=best_threshold)

print("Evaluating on Validation Data...")
val_metrics = evaluate_metrics(y_val, val_probs, threshold=best_threshold)

print("Evaluating on Test Data...")
test_probs = model.predict(X_test_combined, batch_size=batch_size)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)
test_metrics = evaluate_metrics(y_test, test_probs, threshold=best_threshold)

# Function to print metrics neatly
def print_metrics(name, metrics):
    """
    Prints the evaluation metrics.

    Parameters:
    - name: str, name of the dataset (e.g., Training, Validation, Test).
    - metrics: dict, evaluation metrics.
    """
    print(f"\n{name} Metrics:")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"Recall      : {metrics['recall']:.4f}")
    print(f"F1 Score    : {metrics['f1_score']:.4f}")
    print(f"ROC AUC     : {metrics['roc_auc']:.4f}")
    print(f"PR AUC      : {metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

# Print all metrics
print_metrics("Training", train_metrics)
print_metrics("Validation", val_metrics)
print_metrics("Test", test_metrics)

# ==============================
# 10. Save Model and Best Threshold
# ==============================

model_save_path = "residual_fcnn_optimized_tf.h5"
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

# Save the best threshold for future use
threshold_path = "best_threshold.npy"
np.save(threshold_path, best_threshold)
print(f"Best threshold saved to {threshold_path}")
