from pathlib import Path

import pytest
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.self_ask_with_search.prompt import PROMPT
from langchain.llms.loading import load_llm
from steamship import Steamship

from steamship_langchain.llms.openai import OpenAI, OpenAIChat


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
        WIKI_PROMPT.format(input=query, agent_scratchpad=""), stop=["\nObservation:"]
    )
    assert generated.strip().startswith("Thought: ")
    assert generated.strip().endswith("Action: Search[Twins]")


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
        assert len(generation) == 3


@pytest.mark.usefixtures("client")
def test_openai_call(client: Steamship) -> None:
    """Test valid call to openai."""
    llm = OpenAI(client=client, max_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


@pytest.mark.usefixtures("client")
def test_openai_extra_kwargs(client: Steamship) -> None:
    """Test extra kwargs to openai."""
    # Check that foo is saved in extra_kwargs.
    llm = OpenAI(client=client, foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = OpenAI(client=client, foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):  # noqa: PT011
        OpenAI(client=client, foo=3, model_kwargs={"foo": 2})


@pytest.mark.usefixtures("client")
def test_openai_stop_valid(client: Steamship) -> None:
    """Test openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = OpenAI(client=client, stop="3", temperature=0)
    first_output = first_llm(query)
    second_llm = OpenAI(client=client, temperature=0)
    second_output = second_llm(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


@pytest.mark.usefixtures("client")
@pytest.mark.skip()  # Not working yet, loads the wrong OpenAI class
def test_saving_loading_llm(client: Steamship, tmp_path: Path) -> None:
    """Test saving/loading an OpenAPI LLM."""
    llm = OpenAI(client=client, max_tokens=10)
    llm.save(file_path=tmp_path / "openai.yaml")
    loaded_llm = load_llm(tmp_path / "openai.yaml")
    assert loaded_llm == llm


@pytest.mark.usefixtures("client")
def test_openai_streaming_unsupported(client: Steamship) -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(client=client, max_tokens=10)
    with pytest.raises(NotImplementedError):
        llm.stream("I'm Pickle Rick")


@pytest.mark.usefixtures("client")
def test_openai_chat_llm(client: Steamship) -> None:
    """Test Chat version of the LLM"""
    llm = OpenAIChat(client=client)
    llm_result = llm.generate(
        prompts=["Please print the words of the Pledge of Allegiance"], stop=["flag", "Flag"]
    )
    assert len(llm_result.generations) == 1
    generation = llm_result.generations[0]
    assert len(generation) == 1
    text_response = generation[0].text
    assert text_response.strip(' "') == "I pledge allegiance to the"


@pytest.mark.usefixtures("client")
def test_openai_chat_llm_with_prefixed_messages(client: Steamship) -> None:
    """Test Chat version of the LLM"""
    messages = [
        {
            "role": "system",
            "content": "You are EchoGPT. For every prompt you receive, you reply with the exact same text.",
        },
        {"role": "user", "content": "This is a test."},
        {"role": "assistant", "content": "This is a test."},
    ]
    llm = OpenAIChat(client=client, prefix_messages=messages)
    llm_result = llm.generate(prompts=["What is the meaning of life?"])
    assert len(llm_result.generations) == 1
    generation = llm_result.generations[0]
    assert len(generation) == 1
    text_response = generation[0].text
    assert text_response.strip() == "What is the meaning of life?"


@pytest.mark.usefixtures("client")
def test_openai_llm_with_chat_model_init(client: Steamship) -> None:
    """Test Chat version of the LLM, with old init style"""
    messages = [
        {
            "role": "system",
            "content": "You are EchoGPT. For every prompt you receive, you reply with the exact same text.",
        },
        {"role": "user", "content": "This is a test."},
        {"role": "assistant", "content": "This is a test."},
    ]
    llm = OpenAI(client=client, prefix_messages=messages, model_name="gpt-4")
    llm_result = llm.generate(prompts=["What is the meaning of life?"])
    assert len(llm_result.generations) == 1
    generation = llm_result.generations[0]
    assert len(generation) == 1
    text_response = generation[0].text
    assert text_response.strip() == "What is the meaning of life?"
