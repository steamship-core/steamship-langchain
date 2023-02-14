import langchain
from langchain import VectorDBQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from steamship import Block, File, Steamship

from examples.qa_with_sources.client.client import STATE_OF_THE_UNION_PATH
from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import OpenAI
from steamship_langchain.vectorstores import SteamshipVectorStore

client = Steamship(profile="prod")
langchain.llm_cache = SteamshipCache(client)
llm = OpenAI(client=client, temperature=0, cache=True, max_words=250)
index = SteamshipVectorStore(
    client=client, index_name="qa-demo", embedding="text-embedding-ada-002"
)

with STATE_OF_THE_UNION_PATH.open() as f:
    sotu_file = File.create(client, blocks=[Block(text=f.read())])
    file_handle = sotu_file.handle
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
file = File.get(client, handle=file_handle)
texts = [text for block in file.blocks for text in text_splitter.split_text(block.text)]
metadatas = [{"source": f"{file.handle}-offset-{i * 250}"} for i, text in enumerate(texts)]

index.add_texts(texts=texts, metadatas=metadatas)

# new call

import sys

d = sys.version_info[1]

index = SteamshipVectorStore(
    client=client, index_name="qa-demo", embedding="text-embedding-ada-002"
)

ret = index.similarity_search("What did the president say about Justice Breyer?", k=2)
print(ret)

chain = VectorDBQAWithSourcesChain.from_chain_type(
    OpenAI(client=client, temperature=0),
    chain_type="map_reduce",
    vectorstore=index,
)

output = chain({"question": "What did the president say about Justice Breyer?"})
print(output)
