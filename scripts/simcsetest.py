from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_top_k_similar_embeddings(query, embeddings, k):
    similarity_matrix = cosine_similarity(query, embeddings)
    top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]

    # Fix the IndexError
    indices_sorted = top_k_indices[0][np.argsort(-similarity_matrix[0, top_k_indices[0]])]

    return indices_sorted

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

sentences = ['A woman is reading.', 'A man is playing a guitar.', 'The dog barks loudly.', 'Kids are playing soccer in the park.', 'The car is running on the highway.']

# embeddings = model.encode(sentences)
model.build_index(sentences, use_faiss=None, faiss_fast=False, device=None, batch_size=64)

query = "He plays guitar."
result = model.search(query, device=None, threshold=0.0, top_k=3)
print("This is the result")
print(result)