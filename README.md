# Transformer Viewer
Simple visualization for pytorch model, Test version for classification task

# Requirments
python > 3.6  
pytorch > 1.4  
Colr > 0.9  

# How to use
'''python
import torch
import math
import json

class Classifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, num_labels, gpu=False, dropout=0.2):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.gpu = gpu
