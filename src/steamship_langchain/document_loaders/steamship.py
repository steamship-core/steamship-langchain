"""Get LangChain Documents from a set of persistent Steamship Files."""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from pydantic import BaseModel, validator
from steamship import File, Steamship
from steamship.data import TagKind, TagValueKey


def _get_provenance_tag(file: File) -> Optional[str]:
    for tag in file.tags:
        if tag.kind == TagKind.PROVENANCE:
            return tag.value.get(TagValueKey.STRING_VALUE)
    return None


class SteamshipLoader(BaseLoader, BaseModel):
    """Loads Steamship Files into LangChain Documents."""

    client: Steamship
    "Provides Steamship workspace-scoping for File retrieval"

    query: Optional[str] = None
    "Allows free-form query-based retrieval of Files (MAY NOT be used with `files`)." "NOTE: If neither files or query are specified, nothing will be imported."

    files: Optional[List[File]] = None
    "A list of Files to import (MAY NOT be used with `query`)" "NOTE: If neither files or query are specified, nothing will be imported."

    join_str: str = "\n\n"
    "When `collapse_blocks` is True, this determines how the block texts will be joined."

    collapse_blocks: bool = True
    "Determines if all blocks from a File will be merged into a single Document"

    @validator("files", always=True)
    def mutually_exclusive(cls, v, values):  # noqa: N805
        if values.get("query") is not None and v:
            raise ValueError("'query' and 'files' are mutually exclusive.")
        return v

    def load(self) -> List[Document]:
        """Load documents from Steamship Files.

        :raises SteamshipError: can single malformed queries and/or issues with connectivity to backend.
        """
        if self.query:
            source = File.query(self.client, tag_filter_query=self.query).files
        else:
            source = self.files

        docs = []
        for file in source:
            metadata = {"source": file.handle}
            provenance = _get_provenance_tag(file)
            if provenance:
                metadata["provenance"] = provenance

            if self.collapse_blocks:
                text = self.join_str.join([b.text for b in file.blocks])
                docs.append(Document(page_content=text, metadata=metadata))
                continue

            docs.extend([Document(page_content=b.text, metadata=metadata) for b in file.blocks])

        return docs
