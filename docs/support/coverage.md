# Coverage 

To get the most out of Steamship's cloud platform you'll be using adapters. Steamship adapters serve as drop-in replacement for LangChain's modules. Here's an overview of the status of our adapters and their coverage:


## Modules 

* LLMs
  * ✅ LLM Models using `steamship_langchain.llms.OpenAI`
  * ✅ LLM Caching using `steamship_langchain.cache.SteamshipCache`
  * ❌ LLM Serialisation 
  * ✅ Token Usage Tracking supported by `steamship_langchain.llms.OpenAI`
* Callbacks
  * ✅ Log verbose messages via `steamship_langchain.callbacks.LoggingCallbackHandler`
* Document Loaders
  * ✅ Import Steamship Files using `steamship_langchain.document_loaders.SteamshipLoader`
* Prompt
  * ✅ Prompt Templates
  * ✅ Few-shot prompting 
  * ✅ Prompt loading `LangChainHub`
  * ✅ Selecting prompt examples using `ExampleSelector`
* Chains ✅
* Agents 
  * ✅ SERPAPI using `steamship_langchain.tools.SteamshipSERP`
* Memory
  * ✅ ChatMessageHistory using `steamship_langchain.memory.ChatMessageHistory`
* ⚒️ Utils 
  * ✅ VectorStores using `steamship_langchain.vectorstore.SteamshipVectorStore`
  * ✅ Python code splitter using `steamship_langchain.python_splitter.PythonCodeSplitter`
* Steamship File Loaders
  * ✅ Text files: `steamship_langchain.file_loaders.TextFileLoader`
  * ✅ Directories: `steamship_langchain.file_loaders.DirectoryLoader`
  * ✅ GitHub repositories: `steamship_langchain.file_loaders.GitHubRepositoryLoader`
  * ✅ YouTube videos: `steamship_langchain.file_loaders.YouTubeFileLoader`
  * ✅ Various text and image formats: `steamship_langchain.file_loaders.UnstructuredFileLoader`

## Use Cases

* ✅ Chatbot (ChatGPT) 
* ✅ Question Answering with Sources


