import langchain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from steamship import File, Task
from steamship.invocable import PackageService, post

from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import OpenAI


class SummarizeAudioPackage(PackageService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        langchain.llm_cache = SteamshipCache(client=self.client)
        self.llm = OpenAI(client=self.client, cache=True)

    @post("summarize_file")
    def summarize_file(self, file_handle: str) -> str:
        file = File.get(self.client, handle=file_handle)
        text_splitter = CharacterTextSplitter()
        texts = []
        for block in file.blocks:
            texts.extend(text_splitter.split_text(block.text))
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(docs)

    @post("summarize_audio_file")
    def summarize_audio_file(self, file_handle: str) -> Task[str]:
        transcriber = self.client.use_plugin("whisper-s2t-blockifier")
        audio_file = File.get(self.client, handle=file_handle)
        transcribe_task = audio_file.blockify(plugin_instance=transcriber.handle)
        return self.invoke_later(
            "summarize_file",
            wait_on_tasks=[transcribe_task],
            arguments={"file_handle": audio_file.handle},
        )
