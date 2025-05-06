from flask import Flask, request
from flask_cors import CORS
from .embedding import Embedding, get_similarity
from datetime import datetime
import json
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

with open('pdf_chunks.json', 'r') as f:
    all_chunks = json.load(f)
embedding = np.load("embeddings.npy", allow_pickle=True)
embedder = Embedding()

@app.route("/api/chat", methods=['POST'])
def get_papers():
    global all_chunks, embedding, embedder
    message = request.get_json()
    message_content = message["content"]
    message_embedding = embedder.embed_text([message_content])
    best_chunk = {"similarity_score" : 0}
    for i in range(len(all_chunks)):
        curr_chunk = all_chunks[i]
        similarity_score = get_similarity(message_embedding, embedding[i])
        if similarity_score > best_chunk["similarity_score"]:
            curr_chunk["similarity_score"] = similarity_score
            best_chunk = curr_chunk
    return_value = {
        "id": message["id"],
        "role": "assistant",
        "content": json.dumps(best_chunk),
        "timestamp": datetime.now().timestamp(),
    }
    # print(return_value)
    return return_value
