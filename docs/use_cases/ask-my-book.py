from langchain import FAISS, OpenAI, PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain

# Load pages of the book
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings import OpenAIEmbeddings


# Load data
loader = PagedPDFSplitter("../the-almanack-of-naval-ravikant.pdf")
pages = loader.load_and_split()
doc_index = FAISS.from_documents(pages, OpenAIEmbeddings())

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
llm = OpenAI(temperature=0)

doc_chain = load_qa_chain(llm,
                          chain_type="stuff",
                          prompt=QA_PROMPT,
                          verbose=True)
condense_question_chain = LLMChain(
    llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True
)
qa_custom = ChatVectorDBChain(
    vectorstore=doc_index,
    combine_docs_chain=doc_chain,
    question_generator=condense_question_chain,
    verbose=True,
)

# Use chain
chat_history = []

result = qa_custom(
    {"question": "What is specific knowledge", "chat_history": chat_history}
)
chat_history = [(result["question"], result["answer"])]
print("answer 1:", result["answer"])

result = qa_custom(
    {
        "question": """Can you summarise what you just said in bullet points? Formatted as follows: 
        * statement 1
        * statement 2""",
        "chat_history": chat_history,
    }
)
print("answer 2:", result["answer"])
