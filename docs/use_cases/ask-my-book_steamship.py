from time import perf_counter

from langchain import PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
# Load pages of the book
from langchain.chains.question_answering import load_qa_chain
# Load data
from steamship import Steamship

from steamship_langchain import OpenAI
from steamship_langchain.vectorstores import SteamshipVectorStore

client = Steamship()

t0 = perf_counter()
doc_index = SteamshipVectorStore(client=client,
                                 index_name="ask-naval",
                                 embedding="text-embedding-ada-002")
t1 = perf_counter()
# Define prompts
condense_question_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Load Chain

doc_chain = load_qa_chain(OpenAI(client=client, temperature=0, verbose=True),
                          chain_type="stuff",
                          prompt=QA_PROMPT,
                          verbose=True)
question_chain = LLMChain(  # Chain to condense previous input
    llm=OpenAI(client=client, temperature=0, verbose=True),
    prompt=CONDENSE_QUESTION_PROMPT,
)
qa_custom = ChatVectorDBChain(
    vectorstore=doc_index,
    combine_docs_chain=doc_chain,
    question_generator=question_chain,
)

# Use chain
chat_history = []

result = qa_custom(
    {"question": "What is specific knowledge", "chat_history": chat_history}
)
chat_history = [(result["question"], result["answer"])]
print("answer 1:", result["answer"])
print("Loading DB: ", t1 - t0)
print("Test: ", t1 - t1)
print("Total: ", perf_counter() - t0)

# result = qa_custom(
#     {
#         "question": """Can you summarise what you just said in bullet points? Formatted as follows:
#         * statement 1
#         * statement 2""",
#         "chat_history": chat_history,
#     }
# )
# print("answer 2:", result["answer"])
