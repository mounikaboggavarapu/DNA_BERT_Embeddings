# DNA BERT-Based Genomic Sequence Analysis

This repository contains a comprehensive pipeline for analyzing genomic sequences using DNA BERT embeddings and advanced machine learning techniques. The project bridges the gap between bioinformatics and deep learning by leveraging pre-trained language models for DNA sequences and applying cutting-edge classification algorithms.

---

## Table of Contents
1. Introduction
2. Features
3. Project Structure
4. Dependencies
5. Pipeline Overview
6. Results
7. How to Run
8. Future Improvements
9. Acknowledgments

---

## Introduction
Genomic sequence analysis is critical for understanding DNA's biological functions and genetic variations. Traditional methods often lack the power to process massive datasets with high precision. This project introduces a pipeline that uses DNA BERT embeddings to represent DNA sequences and trains machine learning models for classification tasks.

Key Objectives:
- Extract meaningful DNA sequence embeddings using DNA BERT.
- Normalize data and engineer interaction features for improved performance.
- Develop a robust classification model with residual connections to handle complex patterns in genomic data.

---

## Features
- DNA BERT Integration: Generates embeddings for DNA sequences using the Hugging Face DNA BERT model.
- Feature Engineering: Interaction features like element-wise difference and product of embeddings for enhanced classification.
- Custom Models: Residual Fully Connected Neural Network (FCNN) built with PyTorch.
- Cross-Framework Support: Implementation in both PyTorch and TensorFlow.
- Scalable Data Handling: Supports millions of DNA sequences with efficient preprocessing and memory optimization.
- Extensive Evaluation Metrics: Accuracy, F1 Score, Precision, Recall, ROC-AUC, and PR-AUC.

---

## Project Structure

- embeddings_creation.py: Generate DNA BERT embeddings.
- normalization_min_max.py: Normalize embeddings to the range [-1, 1].
- training_pytorch.py: Train and evaluate the model using PyTorch.
- hyper_parameter_tuning.py: Tune model hyperparameters using Keras Tuner.
- kmer.py: Process genomic sequences into k-mers.
- README.md: Project documentation.
- dna_bert_results.txt: Output metrics from the training process.

---


## Key Libraries:
- Python (3.8+)
- PyTorch
- TensorFlow
- Hugging Face Transformers
- NumPy, pandas, scikit-learn
- tqdm (for progress tracking)

---

## Pipeline Overview

1. Embedding Creation
- DNA sequences are processed into 6-mers and passed through the DNA BERT model from Hugging Face.
- Embeddings are extracted for each sequence and saved as .npz files.

2. Data Normalization
- Embeddings are normalized to the range [-1, 1] using min-max scaling.
- The normalized embeddings are split into seq1 and seq2 for subsequent processing.

3. Feature Engineering
- Interaction features are computed:  
  - Element-wise difference: Captures the variance between two sequences.  
  - Element-wise product: Represents shared characteristics.

4. Model Training
- A Residual Fully Connected Neural Network (FCNN) is trained on the combined features.
- Residual connections help retain essential information across layers.

5. Hyperparameter Tuning
- Keras Tuner is used to optimize model parameters (e.g., learning rate, dropout rate, and hidden layer sizes).

6. Evaluation
- Metrics such as accuracy, F1 Score, ROC-AUC, and PR-AUC are computed on the test set.
- Confusion matrix visualization is included to analyze model performance.

---

## Results

Model Performance:
- Accuracy: 77.5%
- Precision: 65.4%
- Recall: 75.3%
- F1 Score: 70.0%
- ROC-AUC: 85.0%
- PR-AUC: 77.4%

Confusion Matrix:
- True Negative: 512
- False Positive: 139
- False Negative: 86
- True Positive: 263

---

## How to Run

1. Clone the Repository:
```bash
git clone https://github.com/your-username/DNA-BERT-Sequence-Analysis.git
cd DNA-BERT-Sequence-Analysis
```

2. Generate Embeddings:
Run the embeddings_creation.py script to generate DNA BERT embeddings:
```bash
python embeddings_creation.py
```

3. Normalize Data:
Normalize the embeddings using:
```bash
python normalization_min_max.py
```

4. Train the Model:
Train the PyTorch-based model with:
```bash
python training_pytorch.py
```

5. (Optional) Tune Hyperparameters:
Run hyper_parameter_tuning.py for optimal hyperparameter selection:
```bash
python hyper_parameter_tuning.py
```

---

