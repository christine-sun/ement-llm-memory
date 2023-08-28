## EMBEDDINGS 1:
# Use spaCy
# Fetching top 3

from memory import Memory
import spacy
from sklearn.neighbors import NearestNeighbors
import openai
from utils import load

spc = spacy.load("en_core_web_md")

class EmbeddingAnswerMatchMemory(Memory):
    def __init__(self, source_text, k=3):
        super().__init__(source_text)
        self.lines = source_text.splitlines()

        # Use spaCy to create Doc objects for all the lines in the file
        line_docs = [spc(line) for line in self.lines]

        # Create a matrix of document vectors
        X = [doc.vector for doc in line_docs]

        # Instantiate a NearestNeighbors object with k=3 (the number of neighbors to return)
        self.nn = NearestNeighbors(n_neighbors=k, algorithm='auto')

        # Fit the NearestNeighbors object to the document vectors
        self.nn.fit(X)

    def query(self, query):

        # Fetch a generic answer to the query
        prompt = f"Answer the question: {query}"
        response1 = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        generic_answer = response1.choices[0].text.strip()

        # Use spaCy to create a Doc object for the new string
        new_doc = spc(generic_answer)

        # Find the indices of the k nearest neighbors to the new_doc
        distances, indices = self.nn.kneighbors([new_doc.vector])

        # Construct the closest embeddings
        closest_embeddings = ""
        for i in range(len(indices[0])):
            closest_embeddings += f"{self.lines[indices[0][i]]}\n"

        # Use the closest embeddings to ask GPT to answer the question
        prompt = f"""You are a smart, knowledgeable, accurate AI with the following information:
            {closest_embeddings}\n
            Please answer the following question: {query}
            """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = EmbeddingMemory(source_text, 3)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)