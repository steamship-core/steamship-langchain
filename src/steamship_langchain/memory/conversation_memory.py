from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains.conversation.base import Memory
from steamship import Block, File, Steamship, SteamshipError, Tag
from steamship.data import TagKind


# copied directly from langchain/langchain/chains/conversation/memory.py
# in repo: https://github.com/hwchase17/langchain
def _get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]


def _timestamp_tag() -> Tag:
    """Return a Tag with the current datetime as the value"""
    return Tag(
        # TODO: TagKind.TIMESTAMP_VALUE
        kind=TagKind.TIMESTAMP,
        value={"timestamp": datetime.now().isoformat()},
    )


def _block_sort_key(block: Block) -> str:
    """Return a sort key for Blocks based on associated timestamp tags."""
    return [
        # TODO: TagKind.TIMESTAMP_VALUE
        tag.value.get("timestamp")
        for tag in block.tags
        if tag.kind == TagKind.TIMESTAMP
    ][0]


class ConversationBufferMemory(Memory):
    """Stores conversations in a Steamship File (instead of a buffer str), providing persistent storage of the
    conversation."""

    client: Steamship
    key: str

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        convo_file = self._get_conversation_file()
        if not convo_file:
            return {self.memory_key: ""}

        blocks = sorted(convo_file.blocks, key=_block_sort_key)
        buffer = "\n".join([block.text for block in blocks])
        return {self.memory_key: buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        block_text = "\n".join([human, ai])

        conversation_file = self._get_or_create_conversation_file()
        Block.create(
            self.client, file_id=conversation_file.id, text=block_text, tags=[_timestamp_tag()]
        )

    def clear(self) -> None:
        """Clear memory contents."""
        self._delete_conversation_file()

    def _get_or_create_conversation_file(self) -> File:
        convo_file = self._get_conversation_file()
        if convo_file:
            return convo_file
        return File.create(self.client, handle=self.key, blocks=[])

    def _get_conversation_file(self) -> Optional[File]:
        try:
            return File.get(self.client, handle=self.key)
        except SteamshipError:
            return None

    def _delete_conversation_file(self):
        convo_file = self._get_conversation_file()
        if convo_file:
            convo_file.delete()


class ConversationBufferWindowMemory(ConversationBufferMemory):
    """Stores conversations in a Steamship File, providing persistent storage of the conversation, returning only the
    last k snippets of the conversation.
    """

    k: int = 5

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        convo_file = self._get_conversation_file()
        if not convo_file:
            return {self.memory_key: ""}

        blocks = sorted(convo_file.blocks, key=_block_sort_key)
        buffer = "\n".join([block.text for block in blocks[-self.k :]])
        return {self.memory_key: buffer}
