"""Demonstration of llm caching with Steamship"""
from time import perf_counter
from typing import Callable

import langchain
from steamship import Steamship

from steamship_langchain import OpenAI
from steamship_langchain.cache import SteamshipCache

client = Steamship()

langchain.llm_cache = SteamshipCache(client=client)


def time(c: Callable, *args, **kwargs):
    t0 = perf_counter()
    c(*args, **kwargs)
    print(perf_counter() - t0)


# Caching = Enabled by default
llm = OpenAI(client=client, model_name="text-davinci-002", n=2, best_of=2)

time(llm, "Tell me a joke")
time(llm, "Tell me a joke")

# Disable caching

print("Disable caching")
llm = OpenAI(client=client, model_name="text-davinci-002", n=2, best_of=2, cache=False)
time(llm, "Tell me a joke")
time(llm, "Tell me a joke")


# Instructions:
#
# 1. First add a steamship client:
#
# from steamship import Steamship
# client = Steamship()
#
# 2. Then replace your cache (InMemoryCache, SQLiteCache, RedisCache, or SQLAlchemyCache) with steamship_langchain.cache.SteamshipCache
