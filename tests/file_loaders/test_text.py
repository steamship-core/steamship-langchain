import pytest
from steamship import Steamship
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders import TextFileLoader
from tests import TEST_ASSETS_PATH

TEST_FILE = "simple.txt"


@pytest.mark.usefixtures("client")
def test_text_loader_no_metadata(client: Steamship):
    path = TEST_ASSETS_PATH / TEST_FILE
    loader_under_test = TextFileLoader(client=client)
    files = loader_under_test.load(str(path))

    assert len(files) == 1

    text_file = files[0]
    assert len(text_file.blocks) == 1
    assert text_file.blocks[0].text == "This is a simple text file."
    assert len(text_file.tags) == 2
    found_timestamp_tag = False
    found_file_tag = False
    for tag in text_file.tags:
        if tag.kind == TagKind.TIMESTAMP:
            found_timestamp_tag = True
            continue
        if (
            tag.kind == TagKind.PROVENANCE
            and tag.name == ProvenanceTag.FILE
            and tag.value.get(TagValueKey.STRING_VALUE).endswith("simple.txt")
        ):
            found_file_tag = True
            continue

    assert found_file_tag
    assert found_timestamp_tag


@pytest.mark.usefixtures("client")
def test_text_loader_with_metadata(client: Steamship):
    path = TEST_ASSETS_PATH / TEST_FILE
    loader_under_test = TextFileLoader(client=client)
    files = loader_under_test.load(str(path), metadata={"import-id": 123456789})

    assert len(files) == 1

    text_file = files[0]
    assert len(text_file.blocks) == 1
    assert text_file.blocks[0].text == "This is a simple text file."
    assert len(text_file.tags) == 3
    found_timestamp_tag = False
    found_file_tag = False
    found_metadata = False
    for tag in text_file.tags:
        if tag.kind == TagKind.TIMESTAMP:
            found_timestamp_tag = True
            continue
        if (
            tag.kind == TagKind.PROVENANCE
            and tag.name == ProvenanceTag.FILE
            and tag.value.get(TagValueKey.STRING_VALUE).endswith("simple.txt")
        ):
            found_file_tag = True
            continue
        if tag.kind == "metadata" and tag.value.get("import-id") == 123456789:
            found_metadata = True
            continue

    assert found_file_tag
    assert found_timestamp_tag
    assert found_metadata
