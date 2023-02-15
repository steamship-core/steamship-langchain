# Steamship Python Client Library For LangChain (🦜️🔗)

[![Steamship](https://raw.githubusercontent.com/steamship-core/python-client/main/badge.svg)](https://www.steamship.com/build/langchain-apps?utm_source=github&utm_medium=badge&utm_campaign=github_repo&utm_id=github_langchain_repo)

[Steamship](https://steamship.com/) is the fastest way to build, ship, and use full-lifecycle language AI.

This repository contains [LangChain](https://langchain.readthedocs.io/en/latest/) adapters for Steamship, enabling 
LangChain developers to rapidly deploy their apps on Steamship to automatically get:

- Production-ready API endpoint(s)
- Horizontal scaling across dependencies / backends
- Persistent storage of app state (including caches)
- Built-in support for Authn/z 
- Multi-tenancy support
- Seamless integration with other Steamship skills (ex: audio transcription) 
- Usage Metrics and Logging
- And more...

Read more about Steamship and LangChain on our [website](https://www.steamship.com/build/langchain-apps?utm_source=github&utm_medium=explainer&utm_campaign=github_repo&utm_id=github_langchain_repo). 

## Installing

Install via pip:

```commandline
pip install steamship-langchain
```

## Adapters

Initial support is offered for the following (with more to follow soon):
- LLMs
  - An adapter is provided for Steamship's OpenAI integration (`steamship_langchain.llms.OpenAI`)
  - An adapter is provided for *caching* LLM calls, via Steamship's Key-Value store (`SteamshipCache`) 
- Tools
  - Search:
    - An adapter is provided for Steamship's SERPAPI integration (`SteamshipSERP`)
- Memory
  - Two adapters that provide persistent conversation memory:
    - Complete Memory (`steamship_langchain.memory.ConversationBufferMemory`)
    - Windowed Memory (`steamship_langchain.memory.ConversationBufferWindowMemory`)

## 📖 Documentation
Please see [here](https://docs.langchain.steamship.com/) for full documentation on:

- Getting started (installation, setting up the environment, simple examples)
- How-To examples (demos, integrations, helper functions)

## Example Use Cases

Here are a few examples of using LangChain on Steamship:
- [Basic Prompting](#basic-prompting)
- [Self Ask With Search](#self-ask-with-search)
- [ChatBot](#chatbot)
- [Summarize Audio](#summarize-audio--async-chaining-)
- [Question Answering With Sources](#question-answering-with-sources--embeddings-)

The examples use temporary workspaces to provide full cleanup during experimentation.
[Workspaces](https://docs.steamship.com/workspaces/index.html) provide a unit of tenant isolation within Steamship.
For production uses, persistent workspaces can be created and retrieved via `Steamship(workspace_handle="my_workspace")` .

> **NOTE**
> These examples omit `import` blocks. Please consult the `examples/` directory for complete source code. 

> **NOTE** 
> Client examples assume that the user has a Steamship API key and that it is exposed to the environment (see: [API Keys](#api-keys))

### Basic Prompting

Example of a basic prompt using a Steamship LLM integration (full source: [examples/greeting](./examples/greeting))

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Simple-LangChain-Prompting-on-Steamship)](https://replit.com/@SteamshipDoug/Simple-LangChain-Prompting-on-Steamship)

#### Server Snippet

```python
from steamship_langchain.llms import OpenAI

@post("greet")
def greet(self, user: str) -> str:
    prompt = PromptTemplate(
      input_variables=["user"],
      template=
      "Create a welcome message for user {user}. Thank them for running their LangChain app on Steamship. "
      "Encourage them to deploy their app via `ship deploy` when ready.",
    )
    llm = OpenAI(client=self.client, temperature=0.8)
    return llm(prompt.format(user=user))
```

#### Client Snippet

```python
with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-app")
    while True:
        name = input("Name: ")
        print(f'{api.invoke("/greet", user=name).strip()}\n')
```

### Self Ask With Search

Executes the LangChain `self-ask-with-search` agent using the Steamship GPT and SERP Tool plugins (full source: [examples/self-ask-with-search](./examples/self-ask-with-search))

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Self-Ask-With-Search-with-LangChain-and-Steamship)](https://replit.com/@SteamshipDoug/Self-Ask-With-Search-with-LangChain-and-Steamship)

#### Server Snippet

```python
from steamship_langchain.llms import OpenAI

@post("/self_ask_with_search")
def self_ask_with_search(self, query: str) -> str:
    llm = OpenAI(client=self.client, temperature=0.0, cache=True)
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

Implements a basic Chatbot (similar to ChatGPT) in Steamship with LangChain (full source: [examples/chatbot](./examples/chatbot)).

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Persistent-ChatBot-with-LangChain-and-Steamship)](https://replit.com/@SteamshipDoug/Persistent-ChatBot-with-LangChain-and-Steamship)

> **NOTE**
> The full ChatBot transcript will persist for the lifetime of the Steamship Workspace. 

#### Server Snippet

```python
from steamship_langchain.llms import OpenAI
from steamship_langchain.memory import ConversationBufferWindowMemory


@post("/send_message")
def send_message(self, message: str, chat_history_handle: str) -> str:
  mem = ConversationBufferWindowMemory(client=self.client,
                                       key=chat_history_handle,
                                       k=2)
  chatgpt = LLMChain(
    llm=OpenAI(client=self.client, temperature=0),
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
        msg = input("You: ")
        print(f"AI: {api.invoke('/send_message', message=msg, chat_history_handle=session_handle)}")
```

### Summarize Audio (Async Chaining)

> [![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#experimental)
>
> Audio transcription support not yet considered fully-production ready on Steamship. We are working hard on
> productionizing support for audio transcription at scale, but there may be some existing issues that you encounter
> as you try this out.


This provides an example of using LangChain to process audio transcriptions
obtained via Steamship's speech-to-text plugins (full source: [examples/summarize-audio](./examples/summarize-audio))

A brief introduction to the Task system (and Task dependencies, for chaining) is
provided in this example. Here, we use `task.wait()` style polling, but time-based
`task.refresh()` style polling, etc., is also available.

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Summarize-Audio-with-LangChain-and-Steamship)](https://replit.com/@SteamshipDoug/Summarize-Audio-with-LangChain-and-Steamship)

#### Server Snippet
```python
from steamship_langchain.llms import OpenAI

@post("summarize_file")
def summarize_file(self, file_handle: str) -> str:
    file = File.get(self.client, handle=file_handle)
    text_splitter = CharacterTextSplitter()
    texts = []
    for block in file.blocks:
        texts.extend(text_splitter.split_text(block.text))
    docs = [Document(page_content=t) for t in texts]
    llm = OpenAI(client=self.client, cache=True)
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
    import_task = File.create_with_plugin(client=client,
                                         plugin_instance=yt_importer.handle, 
                                         url=churchill_yt_url)
    import_task.wait()
    audio_file = import_task.output
    
    summarize_task_response = api.invoke("/summarize_audio_file", audio_file_handle=audio_file.handle)
    summarize_task = Task(client=client, **summarize_task_response)
    summarize_task.wait()
    
    if summarize_task.state == TaskState.succeeded:
      summary = base64.b64decode(summarize_task.output).decode("utf-8")
      print(f"Summary: {summary.strip()}")
```

### Question Answering with Sources (Embeddings)

Provides a basic example of using Steamship to manage embeddings and power a LangChain agent
for question answering with sources (full source: [examples/qa_with_sources](./examples/qa_with_sources))

> **NOTE** 
> The embeddings will persist for the lifetime of the Workspace.

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Question-Answering-with-Sources-using-LangChain-on-Steamship)](https://replit.com/@SteamshipDoug/Question-Answering-with-Sources-using-LangChain-on-Steamship)

#### Server Snippet

```python
from steamship_langchain.llms import OpenAI

def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # set up LLM cache
    langchain.llm_cache = SteamshipCache(self.client)
    # set up LLM
    self.llm = OpenAI(client=self.client,
                            temperature=0,
                            cache=True,
                            max_words=250)
    # create a persistent embedding store
    self.index = self.client.use_plugin(
        "embedding-index",
        config={
            "embedder": {
                "plugin_handle": "openai-embedder",
                "fetch_if_exists": True,
                "config": {
                    "model": "text-embedding-ada-002",
                    "dimensionality": 1536,
                }
            }
        },
        fetch_if_exists=True,
    )

@post("index_file")
def index_file(self, file_handle: str) -> bool:
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    texts = []
    file = File.get(self.client, handle=file_handle)
    for block in file.blocks:
        texts.extend(text_splitter.split_text(block.text))

    # give an approximate source location based on chunk size
    items = [
        Tag(client=self.client,
            text=t,
            value={"source": f"{file.handle}-offset-{i * 250}"})
        for i, t in enumerate(texts)
    ]

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
    chain = load_qa_with_sources_chain(self.llm,
                                       chain_type="map_reduce",
                                       verbose=False)
    search_results = self.search_embeddings(query, k=4)
    docs = [
        Document(page_content=result.tag.text,
                 metadata={"source": result.tag.value.get("source", "unknown")})
        for result in search_results
    ]
    return chain({"input_documents": docs, "question": query})
```

#### Client Snippet

```python
with Steamship.temporary_workspace() as client:
    api = client.use("my-langchain-app")
    
    # Upload the State of the Union address
    with open("state_of_the_union.txt") as f:
        sotu_file = File.create(client, blocks=[Block(text=f.read())])

    # Embed
    api.invoke("/index_file", file_handle=sotu_file.handle)

    # Issue Query
    query = "What did the president say about Justice Breyer?"
    response = api.invoke("/qa_with_sources", query=query)
    print(f"Answer: {response['output_text'].strip()}")
```

## API Keys

Steamship API Keys provide access to our SDK for AI models, including OpenAI, GPT, Cohere, Whisper, and more.

Get your free API key here: https://steamship.com/account/api.

Once you have an API Key, you can :
* Set the env var `STEAMSHIP_API_KEY` for your client
* Pass it directly via `Steamship(api_key=)` or `Steamship.tempory_workspace(api_key=)`.

Alternatively, you can run `ship login`, which will guide you through setting up your environment.

## Deploying on Steamship

Deploying LangChain apps on Steamship is simple: `ship deploy`.

From your package directory (where your `api.py` lives), you can issue the `ship deploy` command to generate a manifest 
file and push your package to Steamship. You may then use the Steamship SDK to create instances of your package in
Workspaces as best fits your needs.

More on deployment and Workspaces can be found in [our docs](https://docs.steamship.com/).

## Feedback and Support

Have any feedback on this package? Or on [Steamship](https://steamship.com) in general?

We'd love to hear from you. Please reach out to: hello@steamship.com, or join us on our [Discord](https://discord.gg/5Vry5ANVwT).
