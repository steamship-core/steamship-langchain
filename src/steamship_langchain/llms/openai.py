import hashlib
import logging
from typing import Any, List, Mapping, Optional

import tiktoken
from langchain.llms.base import BaseLLM, Generation, LLMResult
from pydantic import Field
from steamship import Block, File, Steamship, SteamshipError
from steamship.data import TagKind, TagValueKey


class OpenAI(BaseLLM):
    """Implements LangChain LLM interface in a Steamship-compatible fashion, allowing use in chains/agents as required.

    NOTE: This provides a **synchronous** interaction with the LLM backend.
    """

    client: Steamship
    plugin_handle: str = Field(default="gpt-3", const=True)
    batch_size: int = Field(default=20)
    temperature: float = 0.8
    max_words: int = 256
    n: int = 1
    best_of: int = 1
    batch_task_timeout_seconds: int = 10 * 60  # 10 minute limit on generation tasks

    @property
    def _llm_type(self) -> str:
        return f"steamship-{self.plugin_handle}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "plugin_handle": self.plugin_handle,
            "workspace_handle": self.client.get_workspace().handle,
            "temperature": self.temperature,
            "max_words": self.max_words,
            "cache": self.cache,
            "n": self.n,
            "best_of": self.best_of,
        }

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""

        # steamship requires python 3.8, so tiktoken is fine.
        encoder = "p50k_base"
        enc = tiktoken.get_encoding(encoder)
        tokenized_text = enc.encode(text)
        return len(tokenized_text)

    def _instance_handle(self, stop: Optional[List[str]] = None) -> str:
        params = self._identifying_params
        stop_str = ",".join(stop) if stop is not None else ""
        param_strs = []
        for key, value in params.items():
            param_strs.append(f"{key}-{value!s}")

        param_str = "-".join(param_strs)
        instance_str = f"{param_str}-{stop_str}"
        return f'gpt-{hashlib.sha256(instance_str.encode("utf-8")).hexdigest()}'

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        # TODO(douglas-reid): add validation + stop param checking, etc.

        sub_prompts = [
            prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)
        ]
        generations = []
        for _prompts in sub_prompts:
            sub_generations = self._batch(prompts=_prompts, stop=stop)
            for i in range(0, len(sub_generations), self.n):
                generations.append(sub_generations[i : i + self.n])

        if len(generations) == 0:
            generations = [[Generation(text="Generation failed.")]]

        return LLMResult(generations=generations)

    def _batch(self, prompts: List[str], stop: Optional[List[str]] = None) -> List[Generation]:
        # rudimentary batching implementation

        stop_str = ",".join(stop) if stop is not None else ""
        llm_config = {
            "temperature": self.temperature,
            "max_words": self.max_words,
            "stop": stop_str,
            "n_completions": self.n,
            "best_of": self.best_of,
        }
        instance_handle = self._instance_handle(stop)

        # we create a plugin instance in `_call` because currently `stop` params are passed at configuration-time,
        # not run-time. this will be addressed in planned subsequent versions.
        llm_plugin = self.client.use_plugin(
            plugin_handle=self.plugin_handle,
            instance_handle=instance_handle,
            config=llm_config,
            fetch_if_exists=True,
        )

        blocks = []
        for prompt in prompts:
            blocks.append(Block(text=prompt))

        generations = []
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
                        generation = block_tag.value[TagValueKey.STRING_VALUE]
                        generations.append(Generation(generation))
        except SteamshipError as e:
            logging.error(f"could not generate from OpenAI LLM: {e}")
            # TODO(douglas-reid): determine appropriate action here.
            # for now, if an error is encountered, just swallow.
            pass

        return generations
