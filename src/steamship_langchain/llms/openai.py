import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Dict, Generator, List, Mapping, Optional

from langchain.llms.base import Generation, LLMResult
from langchain.llms.openai import BaseOpenAI
from pydantic import root_validator
from steamship import Block, File, Steamship, SteamshipError
from steamship.data import TagKind, TagValueKey

PLUGIN_HANDLE: str = "gpt-3"
ARGUMENT_WHITELIST = {
    "client",
    "model_name",
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "best_of",
    "model_kwargs",
    "openai_api_key",
    "batch_size",
    "request_timeout",
    # "logit_bias",
    "max_retries",
    "callback_manager",
    "cache",
    "verbose",
}


class OpenAI(BaseOpenAI):
    """Implements LangChain LLM interface in a Steamship-compatible fashion, allowing use in chains/agents as required.

    NOTE: This provides a **synchronous** interaction with the LLM backend.
    """

    client: Steamship  # We can use validate_environment to add the client here
    batch_task_timeout_seconds: int = 10 * 60  # 10 minute limit on generation tasks

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "plugin_handle": PLUGIN_HANDLE,
            "workspace_handle": self.client.get_workspace().handle,
            **super()._identifying_params,
        }

    def _invocation_params(self, stop: Optional[List[str]] = None):
        stop_str = (
            ",".join(stop) if stop is not None and not isinstance(stop, str) else (stop or "")
        )

        return {
            "stop": stop_str,
            **self._default_params,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Steamship's OpenAI Plugin."""
        normal_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_words": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n_completions": self.n,
            "best_of": self.best_of,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            # "logit_bias": self.logit_bias,
        }

        return {**normal_params, **self.model_kwargs}

    @root_validator()
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N805
        return values

    @root_validator(pre=True)
    def add_default_request_timeout(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N805
        values["request_timeout"] = values.get("request_timeout", 600)
        return values

    @root_validator(pre=True)
    def raise_on_unsupported_arguments(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N805

        if unsupported_arguments := set(values.keys()) - ARGUMENT_WHITELIST:
            raise NotImplementedError(f"Found unsupported argument: {unsupported_arguments}")
        return values

    def _instance_handle(self, stop: Optional[List[str]] = None) -> str:
        """Create a unique instance handle based on the invocation params.

        This is required because we don't support runtime parameters yet plugin invocation.
        """
        params = self._invocation_params(stop)
        return (
            f'gpt-{hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()}'
        )

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        # TODO(douglas-reid): add validation + stop param checking, etc.

        sub_prompts = [
            prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)
        ]
        generations = []
        total_token_usage = defaultdict(int)
        for _prompts in sub_prompts:
            sub_generations, token_usage = self._batch(prompts=_prompts, stop=stop)
            for i in range(0, len(sub_generations), self.n):
                generations.append(sub_generations[i : i + self.n])
            for key, usage in token_usage.items():
                total_token_usage[key] += usage

        if len(generations) == 0:
            generations = [[Generation(text="Generation failed.")]]

        return LLMResult(
            generations=generations, llm_output={"token_usage": dict(total_token_usage)}
        )

    def stream(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        raise NotImplementedError("Support for streaming is not supported yet.")

    def completion_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        raise RuntimeError("completion_with_retry is not supported, please use .generate instead.")

    def _batch(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> (List[Generation], Dict[str, int]):
        # rudimentary batching implementation

        llm_config = self._invocation_params(stop)
        instance_handle = self._instance_handle(stop)

        # we create a plugin instance in `_call` because currently `stop` params are passed at configuration-time,
        # not run-time. this will be addressed in planned subsequent versions.
        llm_plugin = self.client.use_plugin(
            plugin_handle=PLUGIN_HANDLE,
            instance_handle=instance_handle,
            config=llm_config,
            fetch_if_exists=True,
        )

        blocks = [Block(text=prompt) for prompt in prompts]

        generations = []
        token_usage = {}
        try:
            prompt_file = File.create(client=self.client, blocks=blocks)
            task = llm_plugin.tag(doc=prompt_file)
            # the llm_plugin handles retries and backoff. this wait()
            # will allow for that to happen.
            task.wait(max_timeout_s=self.batch_task_timeout_seconds)
            generation_file = task.output.file

            for text_block in generation_file.blocks:
                for block_tag in text_block.tags:
                    if block_tag.kind == TagKind.GENERATION:
                        generations.append(
                            Generation(text=block_tag.value[TagValueKey.STRING_VALUE])
                        )

            for file_tag in generation_file.tags:
                if file_tag.kind == "token_usage":
                    token_usage = file_tag.value

        except SteamshipError as e:
            logging.error(f"could not generate from OpenAI LLM: {e}")
            # TODO(douglas-reid): determine appropriate action here.
            # for now, if an error is encountered, just swallow.
            pass

        return generations, token_usage
