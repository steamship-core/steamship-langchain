import pytest
from steamship import Steamship
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders import GitHubRepositoryLoader


@pytest.mark.usefixtures("client")
def test_github_loader(client: Steamship):
    loader_under_test = GitHubRepositoryLoader(
        client=client, repository_path="steamship-core/steamship-langchain", branch_or_tag="main"
    )

    files = loader_under_test.load()
    assert len(files) > 0

    for file in files:
        file.refresh()
        found_tag = False
        for file_tag in file.tags:
            if file_tag.kind == TagKind.PROVENANCE and file_tag.name == ProvenanceTag.URL:
                found_tag = True
                assert (
                    "https://github.com/steamship-core/steamship-langchain"
                    in file_tag.value.get(TagValueKey.STRING_VALUE)
                )
        assert found_tag, f"no file tag found for {file.handle} in [{file.tags}]"
