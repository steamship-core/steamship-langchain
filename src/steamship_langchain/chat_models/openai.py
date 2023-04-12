"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

import tiktoken
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    LLMResult,
    SystemMessage,
)
from pydantic import Extra, Field, root_validator
from steamship import Block, File, MimeTypes, PluginInstance, Steamship, Tag
from steamship.data.tags.tag_constants import RoleTag, TagKind

logger = logging.getLogger(__file__)


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict["content"])
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatOpenAI(BaseChatModel):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    """

    client: Any  #: :meta private:
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    request_timeout: int = 60
    """Timeout in seconds for the OpenAPI request."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    _llm_plugin: PluginInstance

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    def __init__(
        self,
        client: Steamship,
        model_name: str = "gpt-3.5-turbo",
        moderate_output: bool = True,
        **kwargs,
    ):
        super().__init__(client=client, model_name=model_name, **kwargs)
        plugin_config = {"model": self.model_name, "moderate_output": moderate_output}
        if self.openai_api_key:
            plugin_config["openai_api_key"] = self.openai_api_key

        model_args = self.model_kwargs
        for arg in [
            "max_tokens",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "max_retries",
        ]:
            if model_args.get(arg):
                plugin_config[arg] = model_args[arg]

        self._llm_plugin = self.client.use_plugin(
            plugin_handle="gpt-4",
            config=plugin_config,
            fetch_if_exists=True,
        )

    @classmethod
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "temperature": self.temperature,
            # TODO (enias): Add other params
        }

    def completion_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        raise RuntimeError("completion_with_retry is not supported, please use .generate instead.")

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {"model_name": self.model_name}

    def _complete(self, messages: [Dict[str, str]], **params) -> List[BaseMessage]:
        blocks = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if len(content) > 0:
                role_tag = RoleTag(role)
                blocks.append(
                    Block(
                        text=content,
                        tags=[Tag(kind=TagKind.ROLE, name=role_tag)],
                        mime_type=MimeTypes.TXT,
                    )
                )

        file = File.create(self.client, blocks=blocks)
        generate_task = self._llm_plugin.generate(input_file_id=file.id, options=params)
        generate_task.wait()

        return [
            _convert_dict_to_message({"content": block.text, "role": RoleTag.USER.value})
            for block in generate_task.output.blocks
        ]

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        messages = self._complete(messages=message_dicts, **params)
        return ChatResult(
            generations=[ChatGeneration(message=message) for message in messages],
            llm_output={"model_name": self.model_name},
        )

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        raise NotImplementedError("Support for async is not provided yet.")

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{
                "model_name": self.model_name,
                "workspace_handle": self.client.get_workspace().handle,
                "plugin_handle": "gpt-4",
            },
            **self._default_params,
        }

    async def agenerate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        raise NotImplementedError("Support for async is not provided yet.")

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        enc = tiktoken.encoding_for_model(self.model_name)
        tokenized_text = enc.encode(text)
        return len(tokenized_text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""

        model = self.model_name
        if model == "gpt-3.5-turbo":
            # gpt-3.5-turbo may change over time.
            # Returning num tokens assuming gpt-3.5-turbo-0301.
            model = "gpt-3.5-turbo-0301"
        elif model == "gpt-4":
            # gpt-4 may change over time.
            # Returning num tokens assuming gpt-4-0314.
            model = "gpt-4-0314"

        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo-0301":
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens
