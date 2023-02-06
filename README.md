# Steamship Python Client Library For Langchain (🦜️🔗)

Steamship is the fastest way to build, ship, and use full-lifecycle language AI.

This repository contains langchain adapters for Steamship, enabling langchain developers
to rapidly deploy their apps on Steamship to automatically get:

- Production-ready API endpoint(s)
- Horizontal scaling across dependencies / backends
- Persistent storage of app state (including caches)
- Built-in support for Authn/z 
- Multi-tenancy support
- Seamless integration with other Steamship skills (ex: audio transcription) 
- Usage Metrics and Logging
- And much more...

## Installing

Install via pip:

```commandline
pip install steamship-langchain
```

## Examples

Here are a few examples of using langchain on Steamship.

The examples use temporary workspaces to provide full cleanup during experimentation.
[Workspaces](https://docs.steamship.com/workspaces/index.html) provide a unit of tenant isolation within Steamship.
For production uses, persistent workspaces can be created and retrieved via `Steamship(workspace_handle="my_workspace")` .

> **NOTE**
> Thesee examples omit `import` blocks. Please See the `examples/` directory for complete code. 

> **NOTE** 
> Client examples assume that the user has a Steamship API key and that it is exposed to the environment (see: [API Keys](#api-keys))

### Basic Prompting

#### Server Snippet

```python
@post("basic_prompt")
def basic_prompt(self, user: str) -> str:
    prompt = PromptTemplate(
        input_variables=["user"],
        template="Create a welcome message for user {user} and thank them for using langchain on Steamship.",
    )
    llm = SteamshipGPT(client=self.client, temperature=0.8)
    return llm(prompt.format(user=user))
```

#### Client Snippet

```python
with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-app")
    print(api.invoke("/basic_prompt", user="Han Solo"))
```

### Self Ask With Search

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Self-Ask-With-Search-with-LangChain-and-Steamship)](https://replit.com/@SteamshipDoug/Self-Ask-With-Search-with-LangChain-and-Steamship)

#### Server Snippet

```python
@post("/self_ask_with_search")
def self_ask_with_search(self, query: str) -> str:
    llm = SteamshipGPT(client=self.client, temperature=0.0, cache=True)
    serp_tool = SteamshipSERP(client=self.client, cache=True)
    tools = [Tool(name="Intermediate Answer", func=serp_tool.search)]
    self_ask_with_search = initialize_agent(tools, llm, agent="self-ask-with-search", verbose=False)
    return self_ask_with_search.run(query)
```

#### Client Snippet

```python
with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-app")
    query = "Who was president the last time the Twins won the World Series?"
    print(f"Query: {query}")
    print(f"Answer: {api.invoke('/self_ask_with_search', query=query)}")
```

### ChatBot

Implements a basic Chatbot (similar to ChatGPT) in Steamship with LangChain.

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Persistent-ChatBot-with-LangChain-and-Steamship)](https://replit.com/@SteamshipDoug/Persistent-ChatBot-with-LangChain-and-Steamship)

#### Server Snippet

```python
@post("/send_message")
def send_message(self, message: str, chat_history_handle: str) -> str:
    mem = SteamshipPersistentConversationWindowMemory(client=self.client,
                                                      file_handle=chat_history_handle,
                                                      k=2)
    chatgpt = LLMChain(
        llm=SteamshipGPT(client=self.client, temperature=0), 
        prompt=CHATBOT_PROMPT, 
        memory=mem,
    )
    
    return chatgpt.predict(human_input=message)
```

#### Client Snippet

```python
with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-app")
    session_handle = "foo-user-session-1234"
    while True:
        msg = input("> ")
        print(f"{api.invoke('/send_message', message=msg, chat_history_handle=session_handle)}")
```

### Summarize Audio (Async Chaining)

#### Server Snippet
```python

@post("summarize_file")
def summarize_file(self, file_handle: str) -> str:
    file = File.get(self.client, handle=file_handle)
    text_splitter = CharacterTextSplitter()
    texts = []
    for block in file.blocks:
        texts.extend(text_splitter.split_text(block.text))
    docs = [Document(page_content=t) for t in texts]
    llm = SteamshipGPT(client=self.client, cache=True)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

@post("summarize_audio_file")
def summarize_audio_file(self, audio_file_handle: str) -> Task[str]:
    transcriber = self.client.use_plugin("whisper-s2t-blockifier")
    audio_file = File.get(self.client, handle=audio_file_handle)
    transcribe_task = audio_file.blockify(plugin_instance=transcriber.handle)
    return self.invoke_later("summarize_file", wait_on_tasks=[transcribe_task], arguments={"file_handle": audio_file.handle})
```

#### Client Snippet
```python

churchill_yt_url = "https://www.youtube.com/watch?v=MkTw3_PmKtc"

with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-app")
    yt_importer = client.use_plugin("youtube-file-importer")
    audio_file = File.create_with_plugin(client=client,
                                         plugin_instance=yt_importer.handle, 
                                         url=churchill_yt_url)
    
    summarize_task_response = api.invoke("/summarize_audio_file", audio_file_handle=audio_file.handle)
    summarize_task = Task(client=client, **summarize_task_response)
    summarize_task.wait()

    summary = base64.b64decode(summarize_task.output).decode("utf-8")
    print(f"Summary: {summary}")
```

### Question Answering with Sources (Embeddings)

#### Server Snippet

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # create a persistent embedding store
    self.index = self.client.use_plugin(
        "embedding-index",
        config={
            "embedder": {
                "plugin_handle": "openai-embedder",
                "fetch_if_exists": True,
                "config": {
                    "model": "text-similarity-curie-001",
                    "dimensionality": 4096,
                }
            }
        },
        fetch_if_exists=True,
    )
    
@post("embed_file")
def embed_file(self, file_handle: str) -> bool:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = []
    file = File.get(self.client, handle=file_handle)
    for block in file.blocks:
        texts.extend(text_splitter.split_text(block.text))

    items = [Tag(client=self.client, text=t, value={"source": f"{file.handle}-offset-{i*500}"})
             for i, t in enumerate(texts)]

    self.index.insert(items)
    return True

@post("search_embeddings")
def search_embeddings(self, query: str, k: int) -> List[SearchResult]:
    """Return the `k` closest items in the embedding index."""
    search_results = self.index.search(query, k=k)
    search_results.wait()
    items = search_results.output.items
    return items

@post("/qa_with_sources")
def qa_with_sources(self, query: str) -> Dict[str, Any]:
    llm = SteamshipGPT(client=self.client, temperature=0, cache=True)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    search_results = self.search_embeddings(query, k=4)
    docs = [Document(page_content=result.tag.text, metadata={"source": result.tag.value.get("source", "unknown")})
            for result in search_results]
    return chain({"input_documents": docs, "question": query})



```

#### Client Snippet

```python
with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-ap")
    
    # Embed the State of the Union address
    with open("state-of-the-union.txt") as f:
        sotu_file = File.create(self.client, blocks=[Block(text=f.readlines())])
    
    api.invoke("/embed_file", file_handle=sotu_file)

    # Issue Query
    query = "What did the president say about Justice Breyer?"
    print(f"------\nQuery: {query}")
    response = api.invoke('/qa_with_sources', query=query)
    print(f"Answer: {response['output_text']}")

    # Print source
    # NB: assumes a single source is used in response
    last_line = response['output_text'].splitlines()[-1:][0]
    source = last_line[len("SOURCES: "):]
    print(f"------\nSource text ({source}):")
    for input_doc in response['input_documents']:
        metadata = input_doc['metadata']
        src = metadata['source']
        if source == src:
            print(input_doc['page_content'])
            break
```

## API Keys

Steamship API Keys provide access to our SDK for AI models, including OpenAI, GPT, Cohere, and more.

Get your free API key here: https://steamship.com/account/api.

Once you have an API Key, you can :
* Set the env var `STEAMSHIP_API_KEY` for your client
* Pass it directly via `Steamship(api_key=)` or `Steamship.tempory_workspace(api_key=)`.

Alternatively, you can run `ship login`, which will guide you through setting up your environment.