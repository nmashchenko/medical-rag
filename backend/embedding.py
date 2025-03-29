import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity


class Embedding:
    
    def __init__(self, batch_size=8):
        model_version = 'allenai/scibert_scivocab_uncased'
        do_lower_case = True
        self.model = BertModel.from_pretrained(model_version)
        self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
        self.batch_size = batch_size
        
    def embed_text(self, data):
        encoded_inputs = self.tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
        print("Embedded batch size {} chunks".format(len(batch_embeddings)))
        return batch_embeddings

    def create_embeddings(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings.extend(self.embed_text(batch_texts))
        
        np.save('embeddings.npy', embeddings)
        print("Embeddings saved in file \'embeddings.npy\'")
        return embeddings
            
def get_similarity(em1, em2):
    return cosine_similarity(em1.detach().numpy().reshape(1,-1), em2.detach().numpy().reshape(1,-1))

