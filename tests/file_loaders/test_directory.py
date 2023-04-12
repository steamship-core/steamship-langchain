import pytest
from steamship import Steamship
from steamship.data import TagKind
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders import DirectoryLoader, TextFileLoader
from tests import TEST_ASSETS_PATH


@pytest.mark.usefixtures("client")
def test_directory_loader(client: Steamship):
    path = TEST_ASSETS_PATH
    loader_under_test = DirectoryLoader(client=client, file_loader=TextFileLoader(client=client))
    files = loader_under_test.load(str(path), glob="*.*")

    assert len(files) == 3
    for file in files:
        assert len(file.blocks) == 1
        assert len(file.tags) == 2
        found_timestamp_tag = False
        found_file_tag = False
        for tag in file.tags:
            if tag.kind == TagKind.TIMESTAMP:
                found_timestamp_tag = True
                continue
            if tag.kind == TagKind.PROVENANCE and tag.name == ProvenanceTag.FILE:
                found_file_tag = True
                continue

        assert found_file_tag
        assert found_timestamp_tag
