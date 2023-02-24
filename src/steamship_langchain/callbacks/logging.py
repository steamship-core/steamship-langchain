"""Callback Handler that logs with logging package."""
import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs via logging.

    This will allow users to create and access logs of deployed LangChain agents and tools.
    """

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Debug the prompts."""
        logging.debug(f"on_llm_start prompts={json.dumps(prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Debug LLM end"""
        logging.debug("on_llm_end")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Debug new token"""
        logging.debug(f"on_llm_new_token token={token}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        logging.error(f"on_llm_error error={error!s}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log that we are entering a chain."""
        class_name = serialized["name"]
        logging.info(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log that we finished a chain."""
        logging.info("Finished chain.")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        logging.error(f"on_chain_error error={error!s}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        logging.debug(f"on_tool_start input_str={input_str}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        logging.info(f"{action.log}")

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        logging.info(f"{observation_prefix}\n{output}\n{llm_prefix}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        logging.error(f"on_tool_error error={error!s}")

    def on_text(self, text: str, **kwargs: Optional[str]) -> None:
        """Run when agent ends."""
        logging.info(text)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        logging.info(f"{finish.log}")
