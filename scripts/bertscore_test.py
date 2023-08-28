from bert_score import score

# List of reference sentences and candidate sentences
refs = ["Anna is really special to her."]
cands = ["Anna is very special the character because they grew up together."]

# Compute the BERTScore (P, R, and F1)
P, R, F1 = score(cands, refs, lang="en", verbose=True)

# Print BERTScores
print("Precision:", P)
print("Recall:", R)
print("F1 Score:", F1)