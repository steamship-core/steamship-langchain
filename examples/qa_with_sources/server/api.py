from typing import Any, Dict, List

import langchain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from steamship import File, Tag
from steamship.data.plugin.index_plugin_instance import SearchResult
from steamship.invocable import PackageService, post

from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import SteamshipGPT


class QuestionAnsweringPackage(PackageService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # set up LLM cache
        langchain.llm_cache = SteamshipCache(self.client)
        # set up LLM
        self.llm = SteamshipGPT(client=self.client, temperature=0, cache=True, max_words=250)
        # create a persistent embedding store
        self.index = self.client.use_plugin(
            "embedding-index",
            config={
                "embedder": {
                    "plugin_handle": "openai-embedder",
                    "fetch_if_exists": True,
                    "config": {
                        "model": "text-similarity-curie-001",
                        "dimensionality": 4096,
                    },
                }
            },
            fetch_if_exists=True,
        )

    @post("index_file")
    def index_file(self, file_handle: str) -> bool:
        text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        texts = []
        file = File.get(self.client, handle=file_handle)
        for block in file.blocks:
            texts.extend(text_splitter.split_text(block.text))

        # give an approximate source location based on chunk size
        items = [
            Tag(client=self.client, text=t, value={"source": f"{file.handle}-offset-{i * 250}"})
            for i, t in enumerate(texts)
        ]

        self.index.insert(items)
        return True

    @post("search_embeddings")
    def search_embeddings(self, query: str, k: int) -> List[SearchResult]:
        """Return the `k` closest items in the embedding index."""
        search_results = self.index.search(query, k=k)
        search_results.wait()
        items = search_results.output.items
        return items

    @post("/qa_with_sources")
    def qa_with_sources(self, query: str) -> Dict[str, Any]:
        chain = load_qa_with_sources_chain(self.llm, chain_type="map_reduce", verbose=False)
        search_results = self.search_embeddings(query, k=4)
        docs = [
            Document(
                page_content=result.tag.text,
                metadata={"source": result.tag.value.get("source", "unknown")},
            )
            for result in search_results
        ]
        return chain({"input_documents": docs, "question": query})
