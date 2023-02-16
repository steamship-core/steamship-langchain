"""Script to populate the VectorDB."""

from langchain.document_loaders import PagedPDFSplitter
from steamship import Steamship

from steamship_langchain.vectorstores import SteamshipVectorStore

client = Steamship()

loader = PagedPDFSplitter("../the-almanack-of-naval-ravikant.pdf")
pages = loader.load_and_split()
doc_index = SteamshipVectorStore.from_texts(client=client,
                                            texts=[page.page_content for page in pages],
                                            metadatas=[page.metadata for page in pages],
                                            index_name="ask-naval-benchmark",
                                            embedding="text-embedding-ada-002")