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
        """
        Create embeddings of text data

        Parameters:
            data (np.array): np array of strings

        Returns:
            embedding of given data
        """
        encoded_inputs = self.tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
        print("Embedded batch size {} chunks".format(len(batch_embeddings)))
        return batch_embeddings

    def create_embeddings(self, chunks):
        """
        Create embeddings in batches given chunks and saves to .npy file

        Parameters:
            chunks: list of chunk data (dict)

        Returns:
            list of embeddings
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings.extend(self.embed_text(batch_texts))
        
        np.save('embeddings.npy', embeddings)
        print("Embeddings saved in file \'embeddings.npy\'")
        return embeddings
            
def get_similarity(em1, em2):
    """
    Get cosine similarity of two embeddings

    Parameters:
        em1 (torch.Tensor or np.array): First embedding
        em2 (torch.Tensor or np.array): Second embedding

    Returns:
        cosine similarity of em1 and em2
    """
    # convert embeddings to np array
    em1 = em1.detach().numpy() if isinstance(em1, torch.Tensor) else em1
    em2 = em2.detach().numpy() if isinstance(em2, torch.Tensor) else em2
    return float(cosine_similarity(em1.reshape(1,-1), em2.reshape(1,-1))[0][0])

