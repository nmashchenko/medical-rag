from flask import Flask, request
from flask_cors import CORS
from .embedding import Embedding, get_similarity
from datetime import datetime
import json
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

#load paper data and embeddings
with open('pdf_chunks.json', 'r') as f:
    all_chunks = json.load(f)
embedding = np.load("embeddings.npy", allow_pickle=True)
embedder = Embedding()

@app.route("/api/chat", methods=['POST'])
def get_papers():
    global all_chunks, embedding, embedder
    message = request.get_json()
    message_content = message["content"]
    
    # get embedding of query
    message_embedding = embedder.embed_text([message_content])
    best_chunk = {"similarity_score" : 0}
    
    # find chunk with highest similarity score
    for i in range(len(all_chunks)):
        curr_chunk = all_chunks[i]
        similarity_score = get_similarity(message_embedding, embedding[i])
        if similarity_score > best_chunk["similarity_score"]:
            curr_chunk["similarity_score"] = similarity_score
            # print(f"Similarity: {similarity_score}, Source: {curr_chunk['source']}, Index: {curr_chunk['chunk_index']}")
            best_chunk = curr_chunk

    return_value = {
        "id": message["id"],
        "role": "assistant",
        "content": (
            f"Source: http://arxiv.org/pdf/{best_chunk['source'][:-4]}\n"
            f"Similarity: {best_chunk['similarity_score']}\n\n"
            f"{best_chunk['text']}"
        ),
        "timestamp": datetime.now().timestamp(),
    }
    # print(return_value)
    return return_value
