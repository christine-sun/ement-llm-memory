## SUMMARIZATION APPROACH 1:
## Summarize the entire txt corpus at once

from memory import Memory
import openai
from utils import load

class SummarizationFullMemory(Memory):
  def __init__(self, source_text):
    super().__init__(source_text)

    # Fetch the summary of the full text
    prompt = f"""Below is a conversation between a helpful conversational agent and a human.
      \n{source_text[:1790]}
      \nPlease summarize the conversation above. Ensure that the summarization is detailed and includes all relevant information about subjects in the conversation. The summary:
    """
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0,
      max_tokens=250,
    )

    self.summary = response.choices[0].text.strip()

    print("This is the summary!!!! 1")
    print(self.summary)

  def query(self, query):
    prompt = f"""You are a smart, knowledgable, accurate AI with the following information:
      {self.summary}
        \nYou are sure about your answers. Please answer the following question: {query}
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
  memory_test = SummarizationFullMemory(source_text)

  query ="What is Mimi's favorite physical activity?"
  answer = memory_test.query(query)
  print(query)
  print(answer)
