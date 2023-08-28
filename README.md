# EMENT: Enhancing Long-Term Episodic Memory in Large Language Models

This is the source code for EMENT, a research project undertaken with the Princeton University Computer Science department on long-term memory in large language models. 

## Abstract
Current state-of-the-art large language models exhibit poor long-term episodic memory capabilities across extensive token magnitudes. This independent work examines supplementing language models with a memory module during inference to improve its memory capabilities. Five memory module types serve as individual baselines: embeddings, summarization, entity raw sentences, entity summaries, and knowledge graphs. These modules extend beyond existing prompt pre-pending memory techniques and are assessed for their storage and retrieval effectiveness. Baseline evaluations inform memory module hypotheses, guiding the research and development of complex memory modules. This paper’s key insight, the complementary nature of embedding and entity extraction modules, ultimately culminates in the proposal of EMENT, an optimized memory module with statistically significant performance that surpasses current language model agent limitations. This optimized memory module is promising to advance conversational agents’ capabilities, enhancing user experiences across a multitude of applications.

## EMENT (Embeddings + Entity Extraction)
The key finding in this evaluation was that the embedding approaches and entity extraction approaches complemented each other well, where one would pick up on the correct answer where another lacked, and vice versa. This was because in cases where entities failed to be precise (e.g., referring to a subject by their proper noun vs common noun), embeddings were able to supplement the context, essentially “catching” cases where entities failed to be precise.

For example, this is the entity dictionary for the test scenario
> ’TODAY’: ’Hello, how are you doing today?’, ’MIMI’: "YOU: I’m okay, my cat Mimi is sick though. Oh no, why is Mimi sick? YOU: It’s because Mimi loves eating lemons, but she always gets so sick when she eats them. What are you doing to take care of Mimi? YOU: Well, I am just letting Mimi sleep some more. Can you tell me more about Mimi? Mimi is a ginger cat and she likes to swim. When did you first meet Mimi? YOU: I adopted Mimi last May, and she has been such a cuddly companion ever since!", ’LAST MAY’: ’YOU: I adopted Mimi last May, and she has been such a cuddly companion ever since!’

Here, all the information about the human’s cat is captured in the entity “Mimi.” When the
module was queried with “How is the human’s cat doing?”, the entity dictionary failed to find the entity relating to "the human's cat," and therefore on its own, returned the predicted answer: <em>"It is difficult to answer this question without more information. If you could provide more information about the cat, such as its age, breed, and any health issues it may have, I may be able to provide a more accurate answer." </em> However, with even the smallest amount of embeddings (k=3), the embedding memory module was able to fetch the following related sentences for context:
> <em>“AI: That’s too bad. What are you doing to take care of Mimi?”</em> \
> <em>“YOU: Yeah! Mimi is a ginger cat and she likes to swim.”</em> \
> <em>“AI: Oh no, why is Mimi sick?”</em>

which provided just enough context for the module to return “The human’s cat is not doing well.” upon query. This also enabled embeddings to perform better on open-ended questions (e.g. “How is the human doing?”), since the module could fetch a general sentiment from a series of embeddings, as opposed to specific raw sentences in the entity dictionary.
In addition to embeddings catching entities, there were also cases where entities caught embeddings, especially when the returned embeddings were of low quality due to excessive padding around the target information, making it difficult for the embedding library to extract the most relevant information. For example, observe the following question/answer for the Netflix Q42022 letter to shareholders:
> QUESTION: What is the new leadership structure at Netflix?
> ANSWER: Ted Sarandos and Greg Peters are now co-CEOs of Netflix, with Reed Hastings as Executive Chairman.

The top 10 nearest embeddings to the query were as follows:

> 1. Engagement, which is consistent with members on comparable ad- free plans, is better than what we had expected and we believe the lower price point is driving incremental membership growth.
> 2. The silver lining is that the market for entertainment is huge and Netflix is still very small by comparison.
> 3. The quarterly guidance we provide is our actual internal forecast at the time we report.
> 4. Overall the reaction to this launch from both consumers and advertis- ers has confirmed our belief that our ad-supported plan has strong unit economics (at minimum, in-line with or better than the comparable ad-free plan) and will generate incremental revenue and profit, though the impact on 2023 will be modest given that this will build slowly over time.
> 5. For Ryan Murphy’s The Watcher, conversation about the series led to fans flocking to the house, a Saturday Night Live spoof and a spike in home security sales.
> 6. Ted said- “Bela and Scott are outstanding creative executives with proven track records at Netflix.
> 7. 2022’s successful slate was the result of having the best creative executives working with best in class creators.
> 8. Forecast: As we noted in our Q3’22 shareholder letter, revenue is our primary top line metric, particularly as we develop additional revenue streams where membership is just one component of our growth (like advertising and paid sharing).
> 9. Simultaneously, we debuted our Too Hot to Handle game, which has been our biggest game launch to date and is another encouraging sign in the long term opportunity to entertain members with Netflix intellectual property (IP) across different mediums.
> 10. Our Wednesday campaign, with all our marketing in Wednesday’s iconic voice, generated 1.5B organic social impressions pre-launch, a record for a Netflix season one.

With this context, the module returned the following predicted response, which likely used information that came from the language model’s pre-training as opposed to information during inference: <em>The new leadership structure at Netflix is Bela Bajaria as Vice President of Global TV and Scott Stuber as Vice President of Global Film. Both Bela and Scott are creative executives with proven track records at Netflix. </em>(SCORE = 0)
The entity extraction approach with raw sentences, on the other hand, fetched the following context through looking up the “Netflix” entity:
> Ted Sarandos and Greg Peters are now co-CEOs of Netflix, with Reed Hastings as Executive Chairman – completing our succession process. 2022 was a tough year, with a bumpy start but a brighter finish. We believe we have a clear path to reaccelerate our revenue growth- con- tinuing to improve all aspects of Netflix, launching paid sharing and building our ads offering. As always, our north stars remain pleasing our members and building even greater profitability over time. It’s why we’re so focused on continuing to improve our content and broaden our slate. Members watched more Netflix returning seasons and sequels in 2022 than ever before across a broad range of genres – from film (Enola Holmes 2, Lost Bullet 2 and two 365 Days sequels) and unscripted (Sell- ing Sunset S5 and Love is Blind S3) to award winning drama (Ozark S4 and The Crown S5), local language series (Sintonia S3, Young Royals S2 and Alice in Borderland S2, whose success pushed S1 back into the Top 10 after its launch) and animation (Big Mouth S6)...

With this context, the module returned the following predicted response: <em>The new leadership structure at Netflix is Ted Sarandos and Greg Peters as co-CEOs, with Reed Hastings as Executive Chairman.</em> (SCORE = 1)
Overall, in this particular dataset, embeddings top10 had an accuracy of 0.688 and entity raw sentences with prompting had an accuracy of 0.750, while EMENT10 had an accuracy of 0.813.
Ultimately, the recognition of the complementary nature of embeddings and entity extraction (embeddings catch entities and vice versa!) led to the formation of EMENT, which combined the two approaches during storage and retrieval.

### Implementation
#### Storage
1. For every line in source_text, create a spaCy Doc object. Store this as line_docs
2. Create a matrix of document vectors by fetching the corresponding vector for each Doc in
line_docs
3. Instantiate a NearestNeighbors object with k neighbors to return by using the Sklearn’s
NearestNeighbors (an unsupervised learner for implementing neighbor searches)
4. Fit the NearestNeighbors object to the document vectors
5. Initialize an empty entity_dict, which will store all entity → sentence mappings
6. Initialize a Guardrail instance to fix the response from the language model
7. For every line in source_text
Fetch all entities in the line and return it in JSON format using guardrails (see Appendix for the exact prompt)
For every entity in the line’s entities
If the entity exists in entity_dict, append the line to the value associated with it Else, create a new key with this entity in entity_dict and let the value equal this line
8. Store the entity_dict in the memory module

#### Query
1. Create a spaCy Doc object for the query string. Store this as new_doc
2. Find the indices of the k nearest neighbors to the new_doc. distances, indices =
self.nn.kneighbors([new_doc.vector])
3. Fetch the closest embeddings by looping through the indices and concatenating them in
closest_embeddings.
4. Initialize a context string
5. Find all entities in the query
6. For each entity in the query’s entities:
Append the value at the entity (if it exists) to the context string
7. Prompt the language model to fetch an answer given the closest embeddings, context
around the entities, and query. The prompt is
“You are a smart, knowledgeable, accurate AI with the following information: {closest_embeddings} {context}
You are sure about your answers. Please answer the following question: {query}”
8. Return the response

## Performance
The EMENT module outperformed every other memory module in terms of accuracy, even the EMENT module with the smallest amount of embeddings (only k = 3 embeddings). EMENT10 in particular performed the best among all memory modules, with an average accuracy of 0.905 (± 0.06). This outperforms top10 embeddings alone by 15.52% and the best entity extraction approach alone by 49.70%. After conducting a one-sample t-test, we find EMENT10 has a statistically significant better performance than the alternative memory modules.

<strong>EMENT outperformance against other modules</strong>
| Embeddings Top3 | Embeddings Top5 | Embeddings Top10 | Embeddings AnswerMatch | Summarization | Entity Raw SpaCy | Entity Raw Prompt | Entity Summary (by5) SpaCy | Entity Summary (by10) SpaCy | Knowledge Graph | EMENT3 | EMENT5 | EMENT10 |
|-----------------|-----------------|------------------|------------------------|---------------|------------------|-------------------|----------------------------|-----------------------------|-----------------|--------|--------|---------|
| 52.66%          | 33.23%          | 15.52%           | 56.52%                 | 39.90%        | 49.70%           | 75.89%            | 52.79%                     | 45.63%                      | 64.89%          | 30.52% | 12.79% | -       |

## Concluding Thoughts
Ultimately, long-term memory is still an unsolved problem for both humans and artificially intelligent systems. However, through giving language models a hippocampus of their own, we open the possibilities of the applications of memory in language models. Through recognizing the top baseline memory approaches and constructing the optimized EMENT memory module in this independent work, we demonstrate progress in outperforming the initial baseline proposals andsurpassing current industry standards. This research paves the way for further enhancements in memory management within conversational AI agents, ultimately leading to more sophisticated and effective systems. As we look towards the future of memory during inference for language models, promising work can be done in integrating EMENT into a versatile module that can be applied across a wide range of applications, from finance to gaming to education. By expanding EMENT’s applicability and refining the underlying mechanisms, we will be able to facilitate the growth of more advanced AI models boasting enhanced memory capabilities. This, in turn, will enrich human-computer interaction, expand the horizons of AI-driven solutions, and unveil innumerable opportunities for augmenting the human experience itself in the future.
