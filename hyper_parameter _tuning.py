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
import keras_tuner as kt
import gc
import warnings

# ==============================
# 1. Environment Setup
# ==============================

tf.random.set_seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0].name}")
else:
    print("No GPU available, using CPU")

# ==============================
# 2. Data Loading and Preprocessing
# ==============================

train_file_path = "train_normalized_embeddings_minmax_10percent.npz"
test_file_path = "test_normalized_embeddings_minmax.npz"

print("Loading normalized train and test embeddings...")
train_data = np.load(train_file_path)
test_data = np.load(test_file_path)

train_seq1 = train_data['seq1']
train_seq2 = train_data['seq2']
y_train = train_data['labels'].astype(int)

test_seq1 = test_data['seq1']
test_seq2 = test_data['seq2']
y_test = test_data['labels'].astype(int)

print("Computing interaction features...")
difference_train = train_seq1 - train_seq2
product_train = train_seq1 * train_seq2

difference_test = test_seq1 - test_seq2
product_test = test_seq1 * test_seq2

X_train_combined = np.concatenate((train_seq1, train_seq2, difference_train, product_train), axis=1)
X_test_combined = np.concatenate((test_seq1, test_seq2, difference_test, product_test), axis=1)

del train_seq1, train_seq2, difference_train, product_train
del test_seq1, test_seq2, difference_test, product_test
gc.collect()

X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test_combined.shape}")

del X_train_combined
gc.collect()

print("Computing class weights...")
classes = np.unique(y_train_split)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_split)
class_weights_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weights_dict}")

# ==============================
# 3. Define Hypermodel for Tuning
# ==============================

def build_hypermodel(hp):
    input_dim = X_train.shape[1]
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(
        units=hp.Int("units_1", min_value=512, max_value=2048, step=256),
        kernel_regularizer=regularizers.l2(hp.Choice("l2_1", values=[1e-4, 1e-5, 1e-6])),
        activation=None
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(hp.Float("dropout_1", min_value=0.2, max_value=0.5, step=0.1))(x)

    residual = layers.Dense(
        units=hp.Int("units_2", min_value=256, max_value=1024, step=128),
        kernel_regularizer=regularizers.l2(hp.Choice("l2_2", values=[1e-4, 1e-5, 1e-6]))
    )(x)
    residual = layers.BatchNormalization()(residual)

    x_res = layers.Dense(
        units=hp.Int("units_2", min_value=256, max_value=1024, step=128),
        activation="relu",
        kernel_regularizer=regularizers.l2(hp.Choice("l2_2", values=[1e-4, 1e-5, 1e-6]))
    )(x)
    x_res = layers.BatchNormalization()(x_res)
    x_res = layers.Dropout(hp.Float("dropout_2", min_value=0.2, max_value=0.5, step=0.1))(x_res)

    x = layers.Add()([x_res, residual])

    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[1e-3, 1e-4, 5e-5])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==============================
# 4. Hyperparameter Tuning
# ==============================

tuner = kt.Hyperband(
    build_hypermodel,
    objective="val_accuracy",
    max_epochs=30,
    factor=3,
    directory="my_tuning_dir",
    project_name="hyperparameter_tuning"
)

tuner.search(
    X_train,
    y_train_split,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict,
    batch_size=128,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print(f"Units in first layer: {best_hps.get('units_1')}")
print(f"Units in second layer: {best_hps.get('units_2')}")
print(f"L2 regularization: {best_hps.get('l2_1')}")
print(f"Dropout: {best_hps.get('dropout_1')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# ==============================
# 5. Train Final Model
# ==============================

model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train_split,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    class_weight=class_weights_dict,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
)

# ==============================
# 6. Evaluate Model
# ==============================

print("\nEvaluating model...")
test_probs = model.predict(X_test_combined, batch_size=128)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

metrics = {
    'accuracy': accuracy_score(y_test, test_preds),
    'precision': precision_score(y_test, test_preds),
    'recall': recall_score(y_test, test_preds),
    'f1_score': f1_score(y_test, test_preds),
    'roc_auc': roc_auc_score(y_test, test_probs),
    'pr_auc': average_precision_score(y_test, test_probs),
    'confusion_matrix': confusion_matrix(y_test, test_preds)
}

for k, v in metrics.items():
    if k != 'confusion_matrix':
        print(f"{k.capitalize()}: {v:.4f}")
    else:
        print(f"Confusion Matrix:\n{v}")

# ==============================
# 7. Save Model
# ==============================

model_save_path = "residual_fcnn_hyperoptimized_tf.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
