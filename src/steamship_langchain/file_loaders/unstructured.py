"""File loader that uses the `unstructured` package to extract text content."""
from typing import Any, Dict, List, Optional

from pydantic import Field
from steamship import Block, File, MimeTypes

from steamship_langchain.file_loaders import BaseFileLoader, generate_file_tags


class UnstructuredFileLoader(BaseFileLoader):
    """File loader that uses `unstructured` to extract text content prior to upload.

    NOTE: Requires install of unstructured package before use: `pip install unstructured`.
    """

    join_str: Optional[str] = Field(
        default="\n\n",
        description="Determines how text elements in a file will be joined."
        "Set to None to keep elements separate.",
    )

    def __init__(self, **kwargs):
        """Initialize the loader with Steamship workspace and path of file to be loaded.

        :raises ValueError: if `unstructured` package has not been installed.
        """
        super().__init__(**kwargs)
        try:
            import unstructured  # noqa: F401
        except ImportError:
            raise ValueError(
                "failed to import `unstructured`. please install it via `pip install unstructured`"
            )

    def _partition(self, file_path: str) -> List:
        from unstructured.partition.auto import partition

        try:
            return partition(filename=file_path)
        except ValueError:
            return []

    def load(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> List[File]:
        """Load the file into Steamship."""
        tags = generate_file_tags(client=self.client, file_path=path, metadata=metadata)
        elements = self._partition(path)

        if self.join_str:
            text = self.join_str.join([str(element) for element in elements])
            blocks = [Block(text=text)]
        else:
            blocks = [Block(text=str(element)) for element in elements]

        return [File.create(client=self.client, mime_type=MimeTypes.TXT, blocks=blocks, tags=tags)]
