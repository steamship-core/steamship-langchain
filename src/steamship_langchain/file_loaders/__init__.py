"""Provides Importers for Steamship Files.

These importers will put content into Steamship Files that live inside a Steamship workspace. This will provide
persistent storage of the file contents, allowing for repeated testing and access in production applications. Once in a
Steamship workspace, the files can be used in Steamship packages and plugins immediately. This can be useful for a
variety of downstream tasks, including indexing and transcription.
"""

from steamship_langchain.file_loaders.base import (
    BaseFileLoader,
    add_tags_to_file_from_url,
    generate_file_tags,
)
from steamship_langchain.file_loaders.directory import DirectoryLoader
from steamship_langchain.file_loaders.github import GitHubRepositoryLoader
from steamship_langchain.file_loaders.sphinx import (
    ReadTheDocsLoader,
    SphinxSiteLoader,
    SphinxSiteSectionLoader,
)
from steamship_langchain.file_loaders.text import TextFileLoader
from steamship_langchain.file_loaders.unstructured import UnstructuredFileLoader
from steamship_langchain.file_loaders.youtube import YouTubeFileLoader

__all__ = [
    "add_tags_to_file_from_url",
    "BaseFileLoader",
    "DirectoryLoader",
    "generate_file_tags",
    "GitHubRepositoryLoader",
    "ReadTheDocsLoader",
    "SphinxSiteSectionLoader",
    "SphinxSiteLoader",
    "TextFileLoader",
    "UnstructuredFileLoader",
    "YouTubeFileLoader",
]
