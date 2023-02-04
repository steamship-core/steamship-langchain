import pytest
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.self_ask_with_search.prompt import PROMPT
from steamship import Steamship

from steamship_langchain.llms import SteamshipGPT


@pytest.mark.usefixtures("client")
def test_gpt(client: Steamship):
    llm_under_test = SteamshipGPT(client=client, temperature=0)

    # simple prompt
    prompt = "Please respond with a simple 'Hello'"
    generated = llm_under_test(prompt=prompt)
    assert len(generated) != 0
    assert generated.strip() == "Hello"

    query = "Who was the president the first time the Twins won the World Series?"

    # prompt with stop tokens
    generated = llm_under_test(
        PROMPT.format(input=query, agent_scratchpad=""), stop=["Intermediate answer: "]
    )
    assert (
        generated.strip()
        == """Yes.
Follow up: When did the Twins win the World Series for the first time?"""
    )

    # prompt with different stop tokens
    generated = llm_under_test(
        WIKI_PROMPT.format(input=query, agent_scratchpad=""), stop=["\nObservation 1"]
    )
    assert (
        generated.strip()
        == """Thought 1: I need to search Twins and World Series, and find the president the first
time the Twins won the World Series.
Action 1: Search[Twins]"""
    )
