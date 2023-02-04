import pytest
from langchain.schema import Generation
from steamship import Steamship

from steamship_langchain.cache import SteamshipCache

TEST_PROMPT = "this is a test: "
LLM_STRING = "llm"
UNKNOWN = "unknown"


@pytest.mark.usefixtures("client")
def test_cache(client: Steamship):
    cache_under_test = SteamshipCache(client=client)

    cache_value = cache_under_test.lookup(prompt=TEST_PROMPT, llm_string=LLM_STRING)
    assert cache_value is None

    cache_under_test.update(
        prompt=TEST_PROMPT, llm_string=LLM_STRING, return_val=[Generation(text="foo")]
    )
    cache_value = cache_under_test.lookup(prompt=TEST_PROMPT, llm_string=LLM_STRING)
    assert cache_value is not None
    assert cache_value == [Generation(text="foo")]

    unknown_prompt = cache_under_test.lookup(prompt=UNKNOWN, llm_string=LLM_STRING)
    assert unknown_prompt is None

    unknown_llm = cache_under_test.lookup(prompt=TEST_PROMPT, llm_string=UNKNOWN)
    assert unknown_llm is None

    cache_under_test.update(
        prompt=TEST_PROMPT, llm_string=LLM_STRING, return_val=[Generation(text="bar")]
    )
    cache_value = cache_under_test.lookup(prompt=TEST_PROMPT, llm_string=LLM_STRING)
    assert cache_value is not None
    assert cache_value == [Generation(text="bar")]
