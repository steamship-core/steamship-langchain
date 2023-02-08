from typing import Any, Optional

from steamship import File, Steamship, SteamshipError
from steamship.data import TagValueKey
from steamship.utils.kv_store import KeyValueStore


class SteamshipSERP:
    """Provides a Steamship-compatible Search Tool (with optional caching) for use in LangChain chains and agents."""

    client: Steamship
    cache_store: Optional[KeyValueStore] = None

    def __init__(self, client: Steamship, cache: bool = True):
        """Initialize the SteamshipSERP tool.

        This tool uses the serpapi-wrapper plugin. This will use Google searches to provide answers.
        """
        self.client = client
        self.search_tool = self.client.use_plugin("serpapi-wrapper")
        if cache:
            self.cache_store = KeyValueStore(
                client=client, store_identifier="search-tool-serpapi-wrapper"
            )

    def search(self, query: str) -> str:
        """Execute a search using the Steamship plugin."""
        try:
            if self.cache_store is not None:
                value = self.cache_store.get(query)
                if value is not None:
                    return value.get(TagValueKey.STRING_VALUE, "")

            task = self.search_tool.tag(doc=query)
            task.wait()
            answer = self._first_tag_value(
                # TODO: TagKind.SEARCH_RESULT
                task.output.file,
                "search-result",
                TagValueKey.STRING_VALUE,
            )

            if self.cache_store is not None:
                self.cache_store.set(key=query, value={TagValueKey.STRING_VALUE: answer})

            return answer
        except SteamshipError:
            return "No search result found"

    @staticmethod
    def _first_tag_value(file: File, tag_kind: str, value_key: str) -> Optional[Any]:
        """Return the value of the first block tag found in a file for the kind and value_key specified."""
        for block in file.blocks:
            for block_tag in block.tags:
                if block_tag.kind == tag_kind:
                    return block_tag.value.get(value_key, "")
        return None
