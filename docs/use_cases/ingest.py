"""Script to populate the VectorDB."""
import pickle

from langchain import FAISS
from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings import OpenAIEmbeddings

loader = PagedPDFSplitter("../the-almanack-of-naval-ravikant.pdf")
pages = loader.load_and_split()

vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())

# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
