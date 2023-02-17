"""Tests the basics of unstructured for import.

Does not exercise any functionality requiring inference (pdfs, images, etc.).
"""
import tempfile

import pytest
from steamship import Steamship
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders import UnstructuredFileLoader
from tests import TEST_ASSETS_PATH

TEST_TXT_FILE = "simple.txt"
TEST_PYTHON_FILE = "splitter_test_file.py"


@pytest.mark.usefixtures("client")
def test_unstructured_loader_txt_file(client: Steamship):
    # unstructured uses nltk to break text into sentences, etc.
    import nltk

    with tempfile.TemporaryDirectory() as d:
        nltk.download("punkt", download_dir=d)
        nltk.download("averaged_perceptron_tagger", download_dir=d)
        nltk.data.path = [d]

        path = TEST_ASSETS_PATH / TEST_TXT_FILE
        loader_under_test = UnstructuredFileLoader(client=client)
        files = loader_under_test.load(str(path), metadata={"import-id": 123456789})

        assert len(files) == 1

        text_file = files[0]
        assert len(text_file.blocks) == 1
        assert text_file.blocks[0].text == "This is a simple text file."
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


@pytest.mark.usefixtures("client")
def test_unstructured_loader_python_file(client: Steamship):
    # unstructured uses nltk to break text into sentences, etc.
    import nltk

    with tempfile.TemporaryDirectory() as d:
        nltk.download("punkt", download_dir=d)
        nltk.download("averaged_perceptron_tagger", download_dir=d)
        nltk.data.path = [d]

        path = TEST_ASSETS_PATH / TEST_PYTHON_FILE
        loader_under_test = UnstructuredFileLoader(
            client=client, file_path=str(path), join_str="\n"
        )
        files = loader_under_test.load(str(path), metadata={"import-id": 123456789})

        assert len(files) == 1

        py_file = files[0]
        assert len(py_file.blocks) == 1

        found_timestamp_tag = False
        found_file_tag = False
        found_metadata = False
        for tag in py_file.tags:
            if tag.kind == TagKind.TIMESTAMP:
                found_timestamp_tag = True
                continue
            if (
                tag.kind == TagKind.PROVENANCE
                and tag.name == ProvenanceTag.FILE
                and tag.value.get(TagValueKey.STRING_VALUE).endswith(".py")
            ):
                found_file_tag = True
                continue
            if tag.kind == "metadata" and tag.value.get("import-id") == 123456789:
                found_metadata = True
                continue

        assert found_file_tag
        assert found_timestamp_tag
        assert found_metadata
