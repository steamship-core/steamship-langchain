"""Test ChatOpenAI wrapper."""

import pytest
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    SystemMessage,
)
from steamship import Steamship

from steamship_langchain.chat_models.openai import ChatOpenAI


@pytest.mark.usefixtures("client")
def test_chat_openai(client: Steamship) -> None:
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(client=client, max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.usefixtures("client")
def test_chat_openai_system_message(client: Steamship) -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(client=client, max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.usefixtures("client")
def test_chat_openai_generate(client: Steamship) -> None:
    """Test ChatOpenAI wrapper with generate."""
    chat = ChatOpenAI(client=client, max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.usefixtures("client")
def test_chat_openai_multiple_completions(client: Steamship) -> None:
    """Test ChatOpenAI wrapper with multiple completions."""
    chat = ChatOpenAI(client=client, max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.usefixtures("client")
def test_chat_openai_llm_output_contains_model_name(client: Steamship) -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(client=client, max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


@pytest.mark.usefixtures("client")
def test_chat_openai_streaming_llm_output_contains_model_name(client: Steamship) -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(client=client, max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name
