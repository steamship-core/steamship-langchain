"""Import a file from YouTube to Steamship workspace."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from steamship import File, PluginInstance, Steamship, SteamshipError, TaskState

from steamship_langchain.file_loaders import add_tags_to_file_from_url


class YouTubeFileLoader(BaseModel):
    """Load a YouTube file into a Steamship workspace.

    Creates a new `File` by uploading the content of local file. File tags
    for identifying the source file and the time of import are automatically added,
    as well as for any custom metadata that is provided. This enables query-based
    retrieval for downstream processing.

    NOTE: YouTube Files need to be blockified if text-based processing and generation
    is desired. This can be achieved by using Steamship's transcription plugins.
    """

    client: Steamship
    "Provides Steamship workspace-scoping for File loading."

    _yt_importer: PluginInstance = None

    class Config:
        underscore_attrs_are_private = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._yt_importer = self.client.use_plugin("youtube-file-importer")

    def load(self, video_url: str, metadata: Optional[Dict[str, Any]] = None) -> List[File]:
        """Load from file path."""
        task = File.create_with_plugin(
            client=self.client, plugin_instance=self._yt_importer.handle, url=video_url
        )
        while task.state not in [TaskState.failed, TaskState.succeeded]:
            try:
                task.wait(max_timeout_s=60, retry_delay_s=2)
            except SteamshipError as e:
                if not ("timeout" in e.message):
                    raise e

        if task.state == TaskState.failed:
            raise SteamshipError(
                message=f"failed to import youtube video {video_url} : {task.status_message}"
            )

        video_file = task.output

        # because `File.create_with_plugin` doesn't support adding custom tags, we must generate the tags
        # afterwards and add them to the file in a separate step.
        add_tags_to_file_from_url(
            client=self.client, file=video_file, url=video_url, metadata=metadata
        )
        video_file.refresh()

        return [video_file]
