import uuid
from typing import List, Optional, Any, Iterable

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from steamship import Steamship, Tag


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
            index_name: str,
            embedding: OpenAIEmbeddings
    ):
        """Initialize with necessary components."""
        self.client = client
        self.index_name = index_name

        if embedding.document_model_name != embedding.query_model_name:
            raise RuntimeError("SteamshipVectorStore only supports using the same "
                               "OpenAI Embedding model for documents and queries.")

        self.index = client.use_plugin(
            plugin_handle="embedding-index",
            instance_handle=self.index_name,
            config={
                "embedder": {  # TODO: This needs to be derived from embedding_function
                    "plugin_handle": "openai-embedder",
                    "instance_handle": self.index_name,
                    "fetch_if_exists": True,
                    "config": {
                        "model": "text-similarity-curie-001", # TODO: embedding.document_model_name
                        "dimensionality": 4096,
                    },
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
            Tag(client=self.client,
                text=text,
                value=metadata)
            for i, (text, metadata) in enumerate(zip(texts, metadatas))
        ]

        self.index.insert(items)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        search_results = self.index.search(query, k=k)
        search_results.wait()
        return [Document(page_content=item.tag.text, metadata=item.tag.value)
                for item in search_results.output.items]

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
        raise NotImplementedError("Max marginal relevance search not supported yet.")

    @classmethod
    def from_texts(cls, texts: List[str],
                   embedding: Embeddings,  # TODO: Use interface for embedding models
                   metadatas: Optional[List[dict]] = None,
                   **kwargs: Any) -> VectorStore:
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
        client = Steamship()
        index_name = uuid.uuid4().hex
        svs = cls(client=client,
                  index_name=index_name,
                  embedding=embedding)
        svs.add_texts(texts=texts, metadatas=metadatas)
        return svs
