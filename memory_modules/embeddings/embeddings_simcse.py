## EMBEDDINGS 1:
# Use SimCSE
# Fetching top 3

from memory import Memory
from sklearn.neighbors import NearestNeighbors
import openai
from utils import load
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# def get_top_k_similar_embeddings(query, embeddings, k):
#     similarity_matrix = cosine_similarity(query, embeddings)
#     top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]

#     indices_sorted = top_k_indices[0][np.argsort(-similarity_matrix[0, top_k_indices[0]])]

#     return indices_sorted

class EmbeddingTopKSimCSEMemory(Memory):
    def __init__(self, source_text, k=3):
        super().__init__(source_text)

        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

        self.k = k
        self.lines = source_text.splitlines()
        self.model.build_index(self.lines, use_faiss=None, faiss_fast=False, device=None, batch_size=64)
        # self.embeddings = self.model.encode(self.lines)

    def query(self, query):

        # Create query embedding
        # query_embedding = self.model.encode([query])

        # Get the top k similar embeddings
        # top_k_indices = get_top_k_similar_embeddings(query_embedding, self.embeddings, self.k)
        # closest_embeddings = [self.lines[idx] for idx in top_k_indices]

        # Fetch a generic answer to the query
        # generic_answer_prompt = f"Answer the question: {query}"
        # response1 = openai.Completion.create(
        #     engine="text-davinci-003",
        #     prompt=generic_answer_prompt,
        #     temperature=0,
        #     max_tokens=250,
        # )
        # generic_answer = response1.choices[0].text.strip()

        closest_embeddings = self.model.search(query, device=None, threshold=0.0, top_k=self.k)
        print("This is the top k sentences:")
        print(closest_embeddings)

        # Use the closest embeddings to ask GPT to answer the question
        prompt = f"""You are a smart, knowledgeable, accurate AI with the following information:
            {closest_embeddings}\n
            Please answer the following question: {query}
            """
        print("calling gpt3 in embeddings_topk.")
        print(prompt)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = EmbeddingTopKSimCSEMemory(source_text, 3)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)

