import pytest
from steamship import MimeTypes, Steamship
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders import YouTubeFileLoader

# NOTE: This seems dangerous, as this file could go poof! but I don't have better ideas at the moment.
TEST_URL = "https://www.youtube.com/watch?v=MkTw3_PmKtc"


@pytest.mark.usefixtures("client")
@pytest.mark.skip()  # YT loader implementation is failing at the moment due to YT changes.
def test_youtube_loader(client: Steamship):
    loader_under_test = YouTubeFileLoader(client=client)
    files = loader_under_test.load(TEST_URL)

    assert len(files) == 1

    video_file = files[0]
    assert video_file.mime_type == MimeTypes.WEBM_AUDIO
    assert len(video_file.blocks) == 0
    assert len(video_file.tags) == 2
    found_timestamp_tag = False
    found_url_tag = False
    for tag in video_file.tags:
        if tag.kind == TagKind.TIMESTAMP:
            found_timestamp_tag = True
            continue
        if tag.kind == TagKind.PROVENANCE and tag.name == ProvenanceTag.URL:
            found_url_tag = True
            assert TEST_URL in tag.value.get(TagValueKey.STRING_VALUE)
            continue

    assert found_url_tag
    assert found_timestamp_tag
