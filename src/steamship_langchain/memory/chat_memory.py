from datetime import datetime
from typing import List, Optional

from langchain.memory.chat_memory import ChatMessageHistory as BaseChatMessageHistory
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from pydantic import PrivateAttr
from steamship import Block, File, Steamship, SteamshipError, Tag
from steamship.data import TagKind


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


class ChatMessageHistory(BaseChatMessageHistory):
    client: Steamship
    key: str

    HUMAN_PREFIX: str = "Human: "
    AI_PREFIX: str = "AI: "

    _file_handle: str = PrivateAttr()

    def __init__(self, client: Steamship, key: str, *args, **kwargs):
        super().__init__(client=client, key=key, *args, **kwargs)
        self._file_handle = f"history-{self.key}"
        self.messages.extend(
            self.saved_messages
        )  # TODO: is this right? or should this be a prepend?

    @property
    def saved_messages(self) -> List[BaseMessage]:
        file = self._get_conversation_file()
        if not file:
            return []
        blocks = sorted(file.blocks, key=_block_sort_key)

        messages = []
        for b in blocks:
            if b.text.startswith(self.HUMAN_PREFIX):
                messages.append(HumanMessage(content=b.text[len(self.HUMAN_PREFIX) :]))
            elif b.text.startswith(self.AI_PREFIX):
                messages.append(AIMessage(content=b.text[len(self.AI_PREFIX) :]))
            else:
                raise ValueError(f"Found unsupported message type: {b.text}")
        return messages

    def _get_or_create_conversation_file(self) -> File:
        convo_file = self._get_conversation_file()
        if convo_file:
            return convo_file
        return File.create(self.client, handle=self._file_handle, blocks=[])

    def _get_conversation_file(self) -> Optional[File]:
        try:
            return File.get(self.client, handle=self._file_handle)
        except SteamshipError:
            return None

    def _delete_conversation_file(self):
        convo_file = self._get_conversation_file()
        if convo_file:
            convo_file.delete()

    def add_user_message(self, message: str) -> None:
        super().add_user_message(message)
        conversation_file = self._get_or_create_conversation_file()
        Block.create(
            self.client,
            file_id=conversation_file.id,
            text=f"{self.HUMAN_PREFIX}{message}",
            tags=[_timestamp_tag()],
        )

    def add_ai_message(self, message: str) -> None:
        super().add_ai_message(message)
        conversation_file = self._get_or_create_conversation_file()
        Block.create(
            self.client,
            file_id=conversation_file.id,
            text=f"{self.AI_PREFIX}{message}",
            tags=[_timestamp_tag()],
        )

    def clear(self) -> None:
        super().clear()
        self._delete_conversation_file()
