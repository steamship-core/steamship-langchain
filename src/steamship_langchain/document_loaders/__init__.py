"""Provides a Document Loader that loads content from Steamship Files.

With Steamship Files serving as a persistent store, this package provides tooling to translate
content into LangChain documents for use in chains, etc. This functionality can be used for a
variety of tasks, including repeated local testing against a known deployed-state and for deploying
document-processing chains to production environments.
"""

from steamship_langchain.document_loaders.steamship import SteamshipLoader

__all__ = ["SteamshipLoader"]
