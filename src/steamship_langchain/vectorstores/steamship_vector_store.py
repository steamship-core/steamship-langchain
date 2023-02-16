import uuid
from itertools import zip_longest
from typing import Any, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from steamship import Steamship, SteamshipError, Tag

FAMILY_TO_DIMENSIONALITY = {"ada": 1024, "babbage": 2048, "curie": 4096, "davinci": 12288}

MODEL_TO_DIMENSIONALITY = {
    "text-embedding-ada-002": 1536,
    **{
        f"text-similarity-{model}-001": dimensionality
        for model, dimensionality in FAMILY_TO_DIMENSIONALITY.items()
    },
    **{
        f"text-search-{model}-{type}-001": dimensionality
        for type in ["doc", "query"]
        for model, dimensionality in FAMILY_TO_DIMENSIONALITY.items()
    },
    **{
        f"code-search-{model}-{type}-001": FAMILY_TO_DIMENSIONALITY[model]
        for type in ["code", "text"]
        for model in ["babbage", "ada"]
    },
}


def get_dimensionality(model: str) -> int:
    if model not in MODEL_TO_DIMENSIONALITY:
        raise SteamshipError(
            message=f"Model {model} is not supported by this plugin.. "
            + f"Valid models for this task are: {MODEL_TO_DIMENSIONALITY.keys()}."
        )

    return MODEL_TO_DIMENSIONALITY[model]


class SteamshipVectorStore(VectorStore):
    """Wrapper around Steamships vector database.

    Example:
        .. code-block:: python

            from steamship_langchain import SteamshipVectorStore
            faiss = SteamshipVectorStore(embedding_function, index, docstore)

    """

    def __init__(
        self,
        client: Steamship,
        embedding: str,
        index_name: str,
    ):
        """Initialize with necessary components."""

        self.client = client
        self.index_name = index_name or uuid.uuid4().hex

        self.index = client.use_plugin(
            plugin_handle="embedding-index",
            instance_handle=self.index_name,
            config={
                "embedder": {
                    "plugin_handle": "openai-embedder-test",
                    "instance_handle": self.index_name,
                    "fetch_if_exists": False,
                    "config": {"model": embedding, "dimensionality": get_dimensionality(embedding)},
                }
            },
            fetch_if_exists=False,
        )

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None) -> None:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        items = [
            Tag(client=self.client, text=text, value=metadata)
            for i, (text, metadata) in enumerate(zip_longest(texts, metadatas or []))
        ]

        self.index.insert(items)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        search_results = self.index.search(query, k=k)
        search_results.wait()
        return [
            Document(page_content=item.tag.text, metadata=item.tag.value)
            for item in search_results.output.items
        ]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        raise NotImplementedError("Max marginal relevance search not supported yet.")

    @classmethod
    def from_texts(
        cls,
        client: Steamship,
        texts: List[str],
        embedding: str,
        index_name: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Construct SteamshipVectorStore wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the SteamshipVectorStore database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from steamship_langchain import SteamshipVectorStore
                from steamship_langchain.langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                svs = SteamshipVectorStore.from_texts(texts, embeddings)
        """

        svs = cls(client=client, index_name=index_name, embedding=embedding)
        svs.add_texts(texts=texts, metadatas=metadatas)
        return svs
