from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

embeddings = model.encode("A woman is reading.")

# Compute the cosine similarities between two groups of sentences
sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
sentences_b = ['He plays guitar.', 'A woman is making a photo.']
similarities = model.similarity(sentences_a, sentences_b)

# Build index for a group of sentences and search among them
sentences = ['A woman is reading.', 'A man is playing a guitar.']
model.build_index(sentences)
results = model.search("He plays guitar.")