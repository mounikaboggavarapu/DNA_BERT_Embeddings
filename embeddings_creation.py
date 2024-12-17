import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer and model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
bert_model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6")
bert_model.to(device)

print("Freezing DNABERT parameters...")
for param in bert_model.parameters():
    param.requires_grad = False

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t', header=None, names=["seq1", "seq2", "label"])
    print(f"Loaded {len(df)} rows.")
    return df

def get_embeddings(sequences, batch_size=16, max_length=512):
    print(f"Generating embeddings for {len(sequences)} sequences (batch size: {batch_size})...")
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch = sequences[i:i + batch_size]
        encoded_input = tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Load datasets
train_file_path = "dna-6mer-train.txt"
test_file_path = "dna-6mer-test.txt"
train_data = load_data(train_file_path)
test_data = load_data(test_file_path)

# Generate embeddings
print("\nGenerating embeddings for the full train dataset...")
embeddings_seq1_train = get_embeddings(train_data['seq1'].tolist(), batch_size=16)
embeddings_seq2_train = get_embeddings(train_data['seq2'].tolist(), batch_size=16)

print("\nGenerating embeddings for the full test dataset...")
embeddings_seq1_test = get_embeddings(test_data['seq1'].tolist(), batch_size=16)
embeddings_seq2_test = get_embeddings(test_data['seq2'].tolist(), batch_size=16)

# Save embeddings to files
np.savez_compressed("train_embeddings.npz", seq1=embeddings_seq1_train, seq2=embeddings_seq2_train, labels=train_data['label'].values)
np.savez_compressed("test_embeddings.npz", seq1=embeddings_seq1_test, seq2=embeddings_seq2_test, labels=test_data['label'].values)

print("Train embeddings saved to 'train_embeddings.npz'")
print("Test embeddings saved to 'test_embeddings.npz'")
