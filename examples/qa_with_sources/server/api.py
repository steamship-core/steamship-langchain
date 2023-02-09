from typing import Any, Dict, List

import langchain
from langchain import VectorDBQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from steamship import File
from steamship.invocable import PackageService, post

from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import OpenAI
from steamship_langchain.vectorstores import SteamshipVectorStore


class QuestionAnsweringPackage(PackageService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # set up LLM cache
        langchain.llm_cache = SteamshipCache(self.client)
        # set up LLM
        self.llm = OpenAI(client=self.client, temperature=0, cache=True, max_words=250)
        # create a persistent embedding store
        self.index = SteamshipVectorStore(client=self.client, embedding="text-embedding-ada-002")

    @post("index_file")
    def index_file(self, file_handle: str) -> bool:
        text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        file = File.get(self.client, handle=file_handle)
        texts = [text for block in file.blocks for text in text_splitter.split_text(block.text)]
        metadatas = [{"source": f"{file.handle}-offset-{i * 250}"} for i, text in enumerate(texts)]

        self.index.add_texts(texts=texts, metadatas=metadatas)
        return True

    @post("search_embeddings")
    def search_embeddings(self, query: str, k: int) -> List[Document]:
        """Return the `k` closest items in the embedding index."""
        return self.index.similarity_search(query, k=k)

    @post("/qa_with_sources")
    def qa_with_sources(self, query: str) -> Dict[str, Any]:
        chain = VectorDBQAWithSourcesChain.from_chain_type(
            OpenAI(client=self.client, temperature=0),
            chain_type="map_reduce",
            vectorstore=self.index,
        )

        return chain({"question": query})
