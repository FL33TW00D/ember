import requests
import json
import numpy as np

model_id = "sentence-transformers/all-MiniLM-L6-v2"
endpoint = "http://localhost:11434/api/embed"

# This is our "document" corpus
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# Get embeddings for the corpus

def get_payload(documents):
    payload = {
        "model": model_id,
        #"documents": [{"role": "user", "content": d} for d in documents],
        "documents": [{"content": d} for d in documents],
        "options": { "keep_alive": 1 }
    }
    return payload

def get_embeddings(documents):
    payload = get_payload(documents)
    resp = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return np.array(json.loads(resp.content)["embeddings"])

embeddings = get_embeddings(corpus)

# Get embeddings for the query

query = [
    "Someone plays music."
]
query_embeddings = get_embeddings(query)

# Compute scores between query and corpus

scores = np.matmul(query_embeddings, embeddings.T)[0]
max_similarity_index = scores.argmax()

print(f"Most similar entry: {corpus[max_similarity_index]}")

