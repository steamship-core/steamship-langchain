import logging
import tempfile
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from pydantic import Field
from steamship import Steamship
from transformers import GPT2TokenizerFast


class SteamshipGPT(LLM):
    """Implements LangChain LLM interface in a Steamship-compatible fashion, allowing use in chains/agents as required.

    NOTE: This provides a **synchronous** interaction with the LLM backend.
    """

    client: Steamship
    plugin_handle: str = Field(default="gpt-3", const=True)
    temperature: float = 0.8
    max_words: int = 500

    @property
    def _llm_type(self) -> str:
        return f"steamship-{self.plugin_handle}"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        stop_str = ",".join(stop) if stop is not None else ""
        llm_config = {
            "temperature": self.temperature,
            "max_words": self.max_words,
            "stop": stop_str,
        }
        instance_handle = f"gpt-3-{''.join(filter(str.isalnum, stop_str)).lower()}"

        # we create a plugin instance in `_call` because currently `stop` params are passed at configuration-time,
        # not run-time
        llm_plugin = self.client.use_plugin(
            plugin_handle=self.plugin_handle,
            instance_handle=instance_handle,
            config=llm_config,
            fetch_if_exists=True,
        )

        logging.info(f"invoking {self.plugin_handle} with prompt: {prompt}")
        return llm_plugin.generate(prompt=prompt, clean_output=False)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "plugin_handle": self.plugin_handle,
            "workspace_handle": self.client.get_workspace().handle,
            "temperature": self.temperature,
            "max_words": self.max_words,
            "cache": self.cache,
        }

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""

        # create a GPT-3 tokenizer instance
        # NB: as Steamship deploys to AWS Lambda, we must only use the R/W dir of /tmp
        tmp = tempfile.gettempdir()
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=f"{tmp}/tokenizer/")

        # tokenize the text using the GPT-3 tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # calculate the number of tokens in the tokenized text
        return len(tokenized_text)
