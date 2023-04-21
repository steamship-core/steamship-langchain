"""Test ElasticSearch functionality."""

import pytest
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from steamship import Block, File, Steamship, Tag
from steamship.data import TagKind, TagValueKey

from steamship_langchain.vectorstores import SteamshipVectorStore

EMBEDDINGS_MODELS = ["text-embedding-ada-002"]

INDEX_NAME = "test-index-001"


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_from_texts(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SteamshipVectorStore.from_texts(
        client=client, texts=texts, embedding=model, index_name=INDEX_NAME
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_with_metadatas_from_text(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = SteamshipVectorStore.from_texts(
        client=client,
        texts=texts,
        embedding=model,
        metadatas=metadatas,
        index_name=INDEX_NAME,
    )
    output = docsearch.similarity_search("foo", k=1)

    assert output == [Document(page_content="foo", metadata={"page": 0})]


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_add_texts(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = SteamshipVectorStore(client=client, embedding=model, index_name=INDEX_NAME)
    docsearch.add_texts(texts=texts)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_with_metadatas_add_text(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = SteamshipVectorStore(client=client, embedding=model, index_name=INDEX_NAME)
    docsearch.add_texts(texts=texts, metadatas=metadatas)

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_from_files(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""

    first = File.create(
        client=client,
        blocks=[Block(text="Ever since I was a young boy")],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )
    second = File.create(
        client=client,
        blocks=[Block(text="I've played the silver ball")],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )
    third = File.create(
        client=client,
        handle="test-find-me",
        blocks=[Block(text="From Soho down to Brighton"), Block(text="I must have played 'em all")],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )

    svs = SteamshipVectorStore.from_files(
        client=client, embedding=model, index_name=INDEX_NAME, files=[first, second, third]
    )
    output = svs.similarity_search("Brighton", k=1)
    assert output == [
        Document(
            page_content="From Soho down to Brighton",
            metadata={
                "source": "test-find-me-chunk-0",
                "provenance": "pinball-wizard",
            },
        )
    ]


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_add_files(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""

    first = File.create(
        client=client,
        blocks=[Block(text="Ever since I was a young boy")],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )
    second = File.create(
        client=client,
        blocks=[Block(text="I've played the silver ball")],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )
    third = File.create(
        client=client,
        handle="test-find-me",
        blocks=[Block(text="From Soho down to Brighton"), Block(text="I must have played 'em all")],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )

    svs = SteamshipVectorStore(client=client, embedding=model, index_name=INDEX_NAME)
    svs.add_files(files=[first, second, third])
    output = svs.similarity_search("Brighton", k=1)
    assert output == [
        Document(
            page_content="From Soho down to Brighton",
            metadata={
                "source": "test-find-me-chunk-0",
                "provenance": "pinball-wizard",
            },
        )
    ]


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("model", EMBEDDINGS_MODELS)
def test_steamship_vector_store_add_files_with_splitter(client: Steamship, model: str) -> None:
    """Test end to end construction and search."""

    pinball_wizard_lyrics = """Ever since I was a young boy
I've played the silver ball
From Soho down to Brighton
I must have played 'em all
But I ain't seen nothing like him
In any amusement hall
That deaf, dumb and blind kid
Sure plays a mean pinball
He stands like a statue
Becomes part of the machine
Feeling all the bumpers
Always playing clean
He plays by intuition
The digit counters fall
That deaf, dumb and blind kid
Sure plays a mean pinball
He's a pinball wizard
There has got to be a twist
A pinball wizard's
Got such a supple wrist
How do you think he does it? I don't know
What makes him so good?
Ain't got no distractions
Can't hear no buzzers and bells
Don't see no lights a-flashin'
Plays by sense of smell
Always gets a replay
Never seen him fall
That deaf, dumb and blind kid
Sure plays a mean pinball
I thought I was
The Bally table king
But I just handed
My pinball crown to him
Even on my favorite table
He can beat my best
His disciples lead him in
And he just does the rest
He's got crazy flipper fingers
Never seen him fall
That deaf, dumb and blind kid
Sure plays a mean pinball"""

    wizard_file = File.create(
        client=client,
        handle="test-find-me",
        blocks=[Block(text=pinball_wizard_lyrics)],
        tags=[Tag(kind=TagKind.PROVENANCE, value={TagValueKey.STRING_VALUE: "pinball-wizard"})],
    )

    splitter = CharacterTextSplitter(separator="\n", chunk_size=6, chunk_overlap=0)

    svs = SteamshipVectorStore(client=client, embedding=model, index_name=INDEX_NAME)
    svs.add_files(files=[wizard_file], splitter=splitter)
    output = svs.similarity_search("Brighton", k=1)
    assert output == [
        Document(
            page_content="From Soho down to Brighton",
            metadata={
                "source": "test-find-me-chunk-2",
                "provenance": "pinball-wizard",
            },
        )
    ]
