"""Import text files to Steamship workspace."""
from typing import Any, Dict, List, Optional

from steamship import Block, File, MimeTypes

from steamship_langchain.file_loaders import BaseFileLoader, generate_file_tags


class TextFileLoader(BaseFileLoader):
    """Load a text file into a Steamship workspace.

    Creates a new `File` by uploading the content of local file. File tags
    for identifying the source file and the time of import are automatically added,
    as well as for any custom metadata that is provided. This enables query-based
    retrieval for downstream processing.
    """

    def load(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> List[File]:
        """Load from file path."""
        tags = generate_file_tags(client=self.client, file_path=path, metadata=metadata)

        with open(path) as f:
            return [
                File.create(
                    client=self.client,
                    mime_type=MimeTypes.TXT,
                    blocks=[Block(text=f.read())],
                    tags=tags,
                )
            ]
