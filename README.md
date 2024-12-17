# DNA_BERT_Embeddings

embeddings_creation.py : Creating the embedding using the dna_bert hugging face repository.

embeddings_correctness.py : Checking the shapes and embeddings for checking correctness.

embeddings_histogram.py : To check the distribution of embeddings.

normalization_min_max.py : Normalizing the data for the range -1 to 1.

normalization_correctness.py : To check the correctness of normalized data.


training_pytorch.py : Training the model and testing using pytorch.

training_tensorflow.py Training the model and testing using tensorflow

tsne.py : To observe the distribution of normalized data

kmer.py : 3-gram embeddings, normalization, training and testing the model.

Results: DNA_BERT - 77.5 Accuracy
	 3gram - 64.4 Accuracy


Data files used: https://usfedu-my.sharepoint.com/:f:/g/personal/mounikaboggavarapu_usf_edu/EklTf1V-i-5OmHrWbPnx31MB-pFzgS_ueXmcd2LZvPtDlg?e=SFkb8m



Results using pytorch:

Test Set Evaluation Metrics:
Accuracy: 0.7750
Precision: 0.6542
Recall: 0.7536
F1 Score: 0.7004
ROC-AUC: 0.8500
PR-AUC: 0.7737
Confusion Matrix:
[[512 139]
 [ 86 263]]



