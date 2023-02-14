import hashlib
import logging
from typing import Dict, Optional

from langchain.cache import RETURN_VAL_TYPE, BaseCache
from langchain.schema import Generation
from steamship import Steamship
from steamship.utils.kv_store import KeyValueStore


class SteamshipCache(BaseCache):
    """Provide Steamship-compatible caching for LangChain LLM calls."""

    client: Steamship
    key_store_map: Dict[str, KeyValueStore]

    def __init__(self, client: Steamship):
        self.client = client
        self.key_store_map = {}

    @staticmethod
    def _handle_for(llm_string: str) -> str:
        """Generate hash-based ID for a langchain LLM."""
        return f"cache-{hashlib.sha256(llm_string.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _key_for(prompt: str) -> str:
        """Hash prompt to use as key in cache."""
        return f"prompt-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string.

        LangChain uses the `llm_string` to uniquely identify an LLM instance. This cache uses that to generate
        a unique ID for cache storage within a Steamship workspace.
        """
        cache_handle = SteamshipCache._handle_for(llm_string)
        logging.debug(f"cache lookup: {prompt} in {cache_handle}")

        store = self.key_store_map.get(cache_handle) or None
        if store is None:
            store = KeyValueStore(client=self.client, store_identifier=cache_handle)
            self.key_store_map[cache_handle] = store

        value_dict = store.get(key=SteamshipCache._key_for(prompt)) or {}
        if len(value_dict) > 0:
            logging.debug(f"cache hit for {prompt}")
            generations = []
            for _, text in value_dict.items():
                generations.append(Generation(text))
            return generations

        logging.debug(f"cache miss for {prompt}")
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string.

        LangChain uses the `llm_string` to uniquely identify an LLM instance. This cache uses that to generate
        a unique ID for cache storage within a Steamship workspace.
        """

        cache_handle = SteamshipCache._handle_for(llm_string)
        logging.debug(f"cache update for {prompt} in {cache_handle}")

        store = self.key_store_map.get(cache_handle) or None
        if store is None:
            store = KeyValueStore(client=self.client, store_identifier=cache_handle)
            self.key_store_map[cache_handle] = store

        value = {}
        for i, generation in enumerate(return_val):
            value[f"generation-{i}"] = generation.text

        # TODO: should this be synchronous and wait?
        store.set(key=SteamshipCache._key_for(prompt), value=value)
        return None
