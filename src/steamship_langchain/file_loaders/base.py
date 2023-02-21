"""Abstract base class for file loaders."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel
from steamship import File, Steamship, Tag
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag


def generate_file_tags(
    client: Steamship, file_path: str, metadata: Optional[Dict[str, str]] = None
):
    tags = [
        Tag(
            client=client,
            kind=TagKind.TIMESTAMP,
            name="timestamp",
            value={"timestamp": datetime.now().isoformat()},
        ),
        Tag(
            client=client,
            kind=TagKind.PROVENANCE,
            name=ProvenanceTag.FILE,
            value={TagValueKey.STRING_VALUE: file_path},
        ),
    ]
    if metadata:
        tags.append(Tag(client=client, kind="metadata", name="metadata", value=metadata))
    return tags


def add_tags_to_file_from_url(
    client: Steamship, url: str, file: File, metadata: Optional[Dict[str, str]] = None
):
    # todo(douglas-reid): improve interface to be more flexible?
    # todo(douglas-reid): switch to DocTag.TIMESTAMP?

    Tag.create(
        file_id=file.id,
        client=client,
        kind=TagKind.TIMESTAMP,
        name="timestamp",
        value={"timestamp": datetime.now().isoformat()},
    )
    Tag.create(
        file_id=file.id,
        client=client,
        kind=TagKind.PROVENANCE,
        name=ProvenanceTag.URL,
        value={TagValueKey.STRING_VALUE: url},
    )

    if metadata:
        Tag.create(file_id=file.id, client=client, kind="metadata", name="metadata", value=metadata)


class BaseFileLoader(ABC, BaseModel):
    """Base file loader class."""

    client: Steamship
    "Provides Steamship workspace-scoping for Files"

    @abstractmethod
    def load(self, path: str, metadata: Optional[Dict[str, str]] = None) -> List[File]:
        """Load data into Steamship Files."""
