import uuid
from itertools import zip_longest
from typing import Any, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import VectorStore
from steamship import File, Steamship, SteamshipError, Tag
from steamship.data import TagKind, TagValueKey
from steamship.data.plugin.index_plugin_instance import EmbeddingIndexPluginInstance

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


def _get_provenance(file: File) -> str:
    for tag in file.tags:
        if tag.kind == TagKind.PROVENANCE:
            return tag.value.get(TagValueKey.STRING_VALUE) or "unknown"
    return "unknown"


def _sanitize_text(text: str) -> str:
    """Replace known problematic characters for WAF in text with underscores.

    NB: This set is probably woefully incomplete, but should, at the least,
    prevent *known* failures.
    """
    return text.translate(
        str.maketrans(
            {
                "$": "_",
                "%": "_",
            }
        )
    )


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

        self.index: EmbeddingIndexPluginInstance = client.use_plugin(
            plugin_handle="embedding-index",
            instance_handle=self.index_name,
            config={
                "embedder": {
                    "plugin_handle": "openai-embedder",
                    "instance_handle": self.index_name,
                    "fetch_if_exists": True,
                    "config": {"model": embedding, "dimensionality": get_dimensionality(embedding)},
                }
            },
            fetch_if_exists=True,
        )
        self.index.embedder.wait_for_init()

    def add_files(
        self, files: Iterable[File], splitter: Optional[TextSplitter] = None
    ) -> List[str]:
        """Run Steamship Files through the embeddings and add to the vectorstore.

        Args:
            files: collection of Steamship Files to add to the VectorStore
            splitter: optional splitter to use to trim text content of Files into reasonable chunks

        Returns:
            List of ids from adding the files into the VectorStore.
        """
        ids = []
        for file in files:
            source_texts = []
            for block in file.blocks:
                if splitter:
                    source_texts.extend(splitter.split_text(block.text))
                else:
                    source_texts.append(block.text)

            texts = [_sanitize_text(t) for t in source_texts if len(t) > 0]
            if len(texts) > 0:
                provenance = _get_provenance(file)
                metadatas = [
                    {"source": f"{file.handle}-chunk-{i}", "provenance": f"{provenance}"}
                    for i, t in enumerate(texts)
                ]
                ids.extend(self.add_texts(texts, metadatas))

        return ids

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        items = [
            Tag(client=self.client, id=str(uuid.uuid1()), text=text, value=metadata)
            for i, (text, metadata) in enumerate(zip_longest(texts, metadatas or []))
        ]

        self.index.insert(items)

        return [t.id for t in items]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        search_results = self.index.search(query, k=k)
        search_results.wait()
        return [
            Document(page_content=item.tag.text, metadata=item.tag.value)
            for item in search_results.output.items
        ]

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        search_results = self.index.search(query, k=k)
        search_results.wait()
        docs = []
        for item in search_results.output.items:
            doc = Document(page_content=item.tag.text, metadata=item.tag.value)
            docs.append((doc, item.score))
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError("similarity_search_by_vector not supported yet.")

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, **kwargs
    ) -> List[Document]:
        raise NotImplementedError("Max marginal relevance search not supported yet.")

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20, **kwargs
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
        """Construct SteamshipVectorStore wrapper from raw texts.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the SteamshipVectorStore database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from steamship_langchain import SteamshipVectorStore
                svs = SteamshipVectorStore.from_texts(texts, "text-embedding-ada-002", "my-index")
        """

        svs = cls(client=client, index_name=index_name, embedding=embedding)
        svs.add_texts(texts=texts, metadatas=metadatas)
        return svs

    @classmethod
    def from_files(
        cls,
        client: Steamship,
        files: List[File],
        embedding: str,
        index_name: str,
        splitter: Optional[TextSplitter] = None,
    ) -> VectorStore:
        """Construct SteamshipVectorStore wrapper from Steamship Files.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the SteamshipVectorStore database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from steamship_langchain import SteamshipVectorStore
                query = 'filetag and kind "metadata" and name "metadata" and value("import-id")="my-import"'
                files = File.query(client=client, tag_filter_query=query).files
                svs = SteamshipVectorStore.from_files(files, "text-embedding-ada-002", "my-index")
        """

        svs = cls(client=client, index_name=index_name, embedding=embedding)
        svs.add_files(files=files, splitter=splitter)
        return svs
