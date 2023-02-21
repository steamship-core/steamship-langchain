import pytest
from steamship import Block, File, Steamship, SteamshipError

from steamship_langchain.document_loaders import SteamshipLoader
from steamship_langchain.file_loaders import TextFileLoader
from tests import TEST_ASSETS_PATH

TEST_FILE = "simple.txt"


@pytest.mark.usefixtures("client")
def test_steamship_loader_files(client: Steamship):
    path = TEST_ASSETS_PATH / TEST_FILE
    simple_loader = TextFileLoader(client=client)
    files = simple_loader.load(str(path))

    loader_under_test = SteamshipLoader(
        client=client,
        files=files,
    )

    documents = loader_under_test.load()
    assert len(documents) == 1
    assert documents[0].page_content == "This is a simple text file."
    assert documents[0].metadata is not None

    md = documents[0].metadata
    assert md.get("source")
    assert md.get("provenance")


@pytest.mark.usefixtures("client")
def test_steamship_loader_multi_part_files_collapse(client: Steamship):
    test_file = File.create(
        client=client,
        blocks=[
            Block(text="There's a lady who's sure"),
            Block(text="All that glitters is gold"),
            Block(text="And she's buying a stairway to heaven"),
        ],
    )

    # validate collapse
    loader_under_test = SteamshipLoader(
        client=client,
        files=[test_file],
        join_str="\n",
    )

    documents = loader_under_test.load()
    assert len(documents) == 1
    assert (
        documents[0].page_content
        == """There's a lady who's sure
All that glitters is gold
And she's buying a stairway to heaven"""
    )
    assert documents[0].metadata is not None
    md = documents[0].metadata
    assert md.get("source")
    assert md.get("source") == test_file.handle
    assert md.get("provenance") is None


@pytest.mark.usefixtures("client")
def test_steamship_loader_multi_part_files_separate(client: Steamship):
    test_file = File.create(
        client=client,
        blocks=[
            Block(text="There's a lady who's sure"),
            Block(text="All that glitters is gold"),
            Block(text="And she's buying a stairway to heaven"),
        ],
    )

    loader_under_test = SteamshipLoader(client=client, files=[test_file], collapse_blocks=False)

    documents = loader_under_test.load()
    assert len(documents) == 3
    assert documents[0].page_content == "There's a lady who's sure"
    assert documents[1].page_content == "All that glitters is gold"
    assert documents[2].page_content == "And she's buying a stairway to heaven"
    assert documents[0].metadata is not None
    md = documents[0].metadata
    assert md.get("source")
    assert md.get("source") == test_file.handle
    assert md.get("provenance") is None


@pytest.mark.usefixtures("client")
def test_steamship_loader_query(client: Steamship):
    path = TEST_ASSETS_PATH / TEST_FILE
    simple_loader = TextFileLoader(client=client)
    simple_loader.load(str(path))

    loader_under_test = SteamshipLoader(
        client=client, query='filetag and (kind "provenance" and name "file")'
    )

    documents = loader_under_test.load()
    assert len(documents) == 1
    assert documents[0].page_content == "This is a simple text file."
    assert documents[0].metadata is not None

    md = documents[0].metadata
    assert md.get("source")
    assert md.get("provenance")


@pytest.mark.usefixtures("client")
def test_steamship_loader_malformed_query(client: Steamship):
    loader_under_test = SteamshipLoader(client=client, query="footag bar baz")

    with pytest.raises(SteamshipError):
        loader_under_test.load()
