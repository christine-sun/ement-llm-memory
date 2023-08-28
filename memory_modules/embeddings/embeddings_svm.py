## EMBEDDINGS:
# Use SVM (Support vector machine)
# SVM considers the entire cloud of data and it optimizes for
# the hyperplane that "pulls apart" positives from negative
# Prev approach (kNN) doesn't consider the global manifold structure
# and "values" every dimension equally
# SVM finds that way your positive example is unique in the dataset
# and only considers its unique qualities when ranking the
# other examples

from memory import Memory
import spacy
from sklearn import svm
import numpy as np
import openai
from utils import load

spc = spacy.load("en_core_web_md")

class EmbeddingSVMMemory(Memory):
    def __init__(self, source_text, k=3):
        super().__init__(source_text)
        self.lines = source_text.splitlines()
        self.k = k

        # Use spaCy to create Doc objects for all the lines in the file
        line_docs = [spc(line) for line in self.lines]

        # Create a matrix of document vectors
        X = np.array([doc.vector for doc in line_docs])

        # Add a random negative example
        random_negative = np.random.randn(1, X.shape[1])
        X = np.concatenate([X, random_negative], axis=0)

        # Create labels for the data
        y = np.ones(X.shape[0])
        y[-1] = 0  # Set the label of the random negative example to 0

        # Instantiate an SVM object
        self.clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)

        # Fit the SVM object to the document vectors
        self.clf.fit(X, y)

    def query(self, query):

        # Use spaCy to create a Doc object for the new string
        new_doc = spc(query)

        # Infer on the data
        similarities = self.clf.decision_function(np.array([new_doc.vector]))
        sorted_ix = np.argsort(-similarities)

        # Construct the closest embeddings
        closest_embeddings = ""
        for k in sorted_ix[:self.k]:
            closest_embeddings += f"{self.lines[k]}\n"

        # Use the closest embeddings to ask GPT to answer the question
        prompt = f"""You are a smart, knowledgeable, accurate AI with the following information:
            {closest_embeddings}\n
            Please answer the following question: {query}
            """
        print("calling gpt3 in embeddings_svm.")
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
    memory_test = EmbeddingSVMMemory(source_text, 3)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer) # uhh this is not working, im gonna try the other stuff first