import pytest
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.self_ask_with_search.prompt import PROMPT
from steamship import Steamship

from steamship_langchain.llms import OpenAI


@pytest.mark.usefixtures("client")
def test_openai(client: Steamship):
    """Basic tests of the OpenAI plugin wrapper."""
    llm_under_test = OpenAI(client=client, temperature=0)

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
    assert generated.strip().startswith("Thought 1: ")
    assert generated.strip().endswith("Action 1: Search[Twins]")


@pytest.mark.usefixtures("client")
def test_openai_batching(client: Steamship):
    """Basic tests of the OpenAI plugin wrapper batching behavior."""

    # single option generation
    llm_under_test = OpenAI(client=client, temperature=0)

    # batched prompts -- needs to exceed the max batch_size (of 20)
    prompts = ["Tell me a joke", "Tell me a poem"] * 15
    generated = llm_under_test.generate(prompts=prompts)
    assert len(generated.generations) != 0
    assert len(generated.generations) == 30


@pytest.mark.usefixtures("client")
def test_openai_multiple_completions(client: Steamship):
    """Basic tests of the OpenAI plugin wrapper number of completions behavior."""

    llm_under_test = OpenAI(client=client, temperature=0.8, n=3, best_of=3)

    prompts = ["Tell me a joke", "Tell me a poem"] * 5
    generated = llm_under_test.generate(prompts=prompts)
    assert len(generated.generations) != 0
    assert len(generated.generations) == 10
    for generation in generated.generations:
        print(generation)
        assert len(generation) == 3
