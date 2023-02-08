"""Demonstration of llm caching without Steamship"""
from time import perf_counter
from typing import Callable

import langchain
from langchain.cache import InMemoryCache
from langchain.llms import OpenAI

langchain.llm_cache = InMemoryCache()


def time(callable: Callable, *args, **kwargs):
    t0 = perf_counter()
    callable(*args, **kwargs)
    print(perf_counter() - t0)


# Caching = Enabled by default
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

time(llm, "Tell me a joke")
time(llm, "Tell me a joke")

# Disable caching

print("Disable caching")
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=False)
time(llm, "Tell me a joke")
time(llm, "Tell me a joke")
