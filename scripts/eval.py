from utils import load, load_json
import json
import openai
import time
from bert_score import score
import numpy as np
from simcse import SimCSE
import os

# === MEMORY TYPES ===
# Are you adding a new memory type?
# 1 - Import it here
from embeddings_topk import EmbeddingTopKMemory
from embeddings_simcse import EmbeddingTopKSimCSEMemory
from embeddings_answermatch import EmbeddingAnswerMatchMemory
from summarization_full import SummarizationFullMemory
from summarization_byline import SummarizationByLineMemory
from entity_rawspacy import EntityRawSpaCyMemory
from entity_rawprompt import EntityRawPromptMemory
from entity_summaryspacy import EntitySummarySpaCyMemory
from kgraph import KnowledgeGraphMemory
from embeddings_entity import EmbeddingEntityMemory

# 2 - Add it to the dictionary here
memories = {
    "embeddings_topk": EmbeddingTopKMemory,
    "embeddings_simcse": EmbeddingTopKSimCSEMemory,
    "embeddings_answermatch": EmbeddingAnswerMatchMemory,
    "summarization_full": SummarizationFullMemory,
    "summarization_byline": SummarizationByLineMemory,
    "entity_rawspacy": EntityRawSpaCyMemory,
    "entity_rawprompt": EntityRawPromptMemory,
    "entity_summaryspacy": EntitySummarySpaCyMemory,
    "kgraph": KnowledgeGraphMemory,
    "embeddings_entity": EmbeddingEntityMemory,
}
# 3 - Add its corresponding arguments here
args = {
    "embeddings_topk": {"k": 3},
    "embeddings_topk": [{"k": 3}, {"k": 5}, {"k": 10}],
    "embeddings_simcse": [{"k": 3}, {"k": 5}, {"k": 10}],
    "embeddings_answermatch": [{"k": 10}],
    "summarization_full": [
        # means there are no args
        {"k": -1}
    ],
    # "summarization_byline": [
    #     {"k": 5},
    #     {"k": 10},
    # ],
    "entity_rawspacy": [
        # means there are no args
        {"k": -1}
    ],
    "entity_rawprompt": [{"k": 5}],
    "entity_summaryspacy": [
        {"k": 5},
        {"k": 10},
    ],
    "kgraph": [{"k": -2}],
    "embeddings_entity": [{"k": 3}, {"k": 5}, {"k": 10}],
}

# ===
# Are you adding a new test file? Put the id here
files = ["netflix_q42022"]

predicted_answers = {}

# === SCORE PREDICTED ANSWERS ===
def score_answer(correct_answer, predicted_answer):
    # Use GPT to score answers
    prompt = f"""Given the correct answer: {correct_answer}
        and a predicted answer: {predicted_answer},
        return 1 if the predicted answer is mostly correct and 0 only if it is completely incorrect. If the correct answer is a Yes/No question, only look at the Yes/No part of the predicted answer."""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=3,
        n=1,
        temperature=0,
    )
    gpt_score = int(response.choices[0].text.strip())
    return gpt_score


# === FETCH AND STORE PREDICTED ANSWERS AND THEIR SCORES ===

# Evaluate for all files (tests)
for file_id in files:
    print(f"== Starting eval on file {file_id} ==")
    # Load all of the test files
    source_text = load(file_id + ".txt")

    # Get its corresponding questions
    file_json = file_id + "_questions.json"
    all_questions = load_json(file_json)

    # For storing question_type -> predicted_answer/score
    predicted_answers[file_id] = {}

    # Test each Memory against given file
    for memory_key, memory_class in memories.items():
        print(f"= Starting eval on memory type {memory_key} =")
        t0 = time.time()

        # Iterate through all testing argument values
        for sub_args in args[memory_key]:

            # Create new memory_key name depending on if we are varying the arguments
            arg_key, arg_value = list(sub_args.items())[0]
            if arg_value == -1:
                new_memory_key = memory_key
            else:
                new_memory_key = "" + memory_key + "_" + arg_key + "=" + str(arg_value)

            # Create this Memory type with this file and corresponding arguments
            if arg_value == -1:
                memory = memory_class(*[source_text])
            # Accomodate for knowledge graph module - TODO: more elegant way?
            elif arg_value == -2:
                memory = memory_class(*[source_text], file_id)
            else:
                memory = memory_class(*[source_text], arg_value)

            # Create predicted answers section for this Memory
            predicted_answers[file_id][new_memory_key] = {}
            predicted_answers[file_id][new_memory_key]["evaluations"] = {}
            predicted_answers[file_id][new_memory_key]["evaluations"]["total"] = {
                "sum": 0,
                "count": 0,
                "average": 0,
            }

            # Test out the Memory on each question
            for question_type, questions in all_questions.items():

                # Create predicted answers section for this question type
                predicted_answers[file_id][new_memory_key][question_type] = []
                predicted_answers[file_id][new_memory_key]["evaluations"][
                    question_type
                ] = {"sum": 0, "count": 0, "average": 0}

                for question in questions:
                    # Query the Memory with the question
                    predicted_answer = memory.query(question["question"])

                    # Determine the correctness of the answer
                    predicted_score = score_answer(question["answer"], predicted_answer)

                    # Update the sum and count of this question type
                    predicted_answers[file_id][new_memory_key]["evaluations"][
                        question_type
                    ]["sum"] += predicted_score
                    predicted_answers[file_id][new_memory_key]["evaluations"][
                        question_type
                    ]["count"] += 1

                    # Save the predicted answer and its score under the file_id, memory_type, and question_type
                    predicted_answers[file_id][new_memory_key][question_type].append(
                        {"answer": predicted_answer, "score": predicted_score}
                    )

                # Compute average for this question type
                curr_sum = float(
                    predicted_answers[file_id][new_memory_key]["evaluations"][
                        question_type
                    ]["sum"]
                )
                curr_count = float(
                    predicted_answers[file_id][new_memory_key]["evaluations"][
                        question_type
                    ]["count"]
                )
                predicted_answers[file_id][new_memory_key]["evaluations"][
                    question_type
                ]["average"] = (curr_sum / curr_count)

                # Find total sum and count
                predicted_answers[file_id][new_memory_key]["evaluations"]["total"][
                    "sum"
                ] += curr_sum
                predicted_answers[file_id][new_memory_key]["evaluations"]["total"][
                    "count"
                ] += curr_count

            # Compute average across all questions
            total_sum = float(
                predicted_answers[file_id][new_memory_key]["evaluations"]["total"][
                    "sum"
                ]
            )
            total_count = float(
                predicted_answers[file_id][new_memory_key]["evaluations"]["total"][
                    "count"
                ]
            )
            predicted_answers[file_id][new_memory_key]["evaluations"]["total"][
                "average"
            ] = (total_sum / total_count)

            # Compute time for this memory type
            t1 = time.time()
            predicted_answers[file_id][new_memory_key]["evaluations"]["total"][
                "time"
            ] = (t1 - t0)

    # Write predicted answers to fileid_predicted_answers.json
    with open(f"{file_id}_predicted_answers.json", "w") as f:
        json.dump(predicted_answers, f)
