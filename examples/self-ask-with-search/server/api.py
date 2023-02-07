from typing import Any, Dict

import langchain
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from steamship.invocable import PackageService, post

from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import OpenAI
from steamship_langchain.tools import SteamshipSERP


class SelfAskWithSeachPackage(PackageService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Sets up the langchain global cache for LLM calls
        langchain.llm_cache = SteamshipCache(client=self.client)

    @post("/self_ask_with_search")
    def self_ask_with_search(self, query: str) -> Dict[str, Any]:
        """Returns a dictionary containing both the answer for the query and any intermediate steps taken."""
        llm = OpenAI(client=self.client, temperature=0.0, cache=True)
        serp_tool = SteamshipSERP(client=self.client, cache=True)
        tools = [Tool(name="Intermediate Answer", func=serp_tool.search)]
        self_ask_with_search = initialize_agent(
            tools, llm, agent="self-ask-with-search", verbose=False, return_intermediate_steps=True
        )
        return self_ask_with_search(query)
