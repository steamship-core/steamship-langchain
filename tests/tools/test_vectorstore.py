"""Test ElasticSearch functionality."""
import pytest
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from steamship import Steamship

from steamship_langchain.vectorstores import SteamshipVectorStore
from tests.tools.fake_embeddings import FakeEmbeddings


@pytest.mark.usefixtures("client")
def test_steamship_vector_store_from_texts(client: Steamship) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SteamshipVectorStore.from_texts(
        client=client, texts=texts, embedding=OpenAIEmbeddings()
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.usefixtures("client")
def test_steamship_vector_store_with_metadatas_from_text(client: Steamship) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = SteamshipVectorStore.from_texts(
        client=client, texts=texts, embedding=OpenAIEmbeddings(), metadatas=metadatas
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


@pytest.mark.usefixtures("client")
def test_steamship_vector_store_add_texts(client: Steamship) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SteamshipVectorStore(client=client, embedding=OpenAIEmbeddings())
    docsearch.add_texts(texts=texts)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.usefixtures("client")
def test_steamship_vector_store_with_metadatas_add_text(client: Steamship) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = SteamshipVectorStore(
        client=client,
        embedding=OpenAIEmbeddings(),
    )
    docsearch.add_texts(texts=texts, metadatas=metadatas)

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


@pytest.mark.usefixtures("client")
def test_steamship_vector_store_non_openai_embeddings(client: Steamship) -> None:
    """Test adding of texts to a docstore that doesn't support it."""
    with pytest.raises(NotImplementedError):
        _ = SteamshipVectorStore(client=client, index_name=None, embedding=FakeEmbeddings())
