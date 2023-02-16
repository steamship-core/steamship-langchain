# Coverage 

To get the most out of Steamship's cloud platform you'll be using adapters. Steamship adapters serve as drop-in replacement for LangChain's modules. Here's an overview of the status of our adapters and their coverage:


## Modules 

* LLMs
  * ✅ LLM Models using `steamship_langchain.llms.OpenAI`
  * ✅ LLM Caching using `steamship_langchain.cache.SteamshipCache`
  * ❌ LLM Serialisation 
  * ✅ Token Usage Tracking supported by `steamship_langchain.llms.OpenAI`
* Prompt ✅
  * ✅ Prompt Templates
  * ✅ Few-shot prompting 
  * ✅ Prompt loading `LangChainHub`
  * ✅ Selecting prompt examples using `ExampleSelector`
* Chains ✅
* Agents 
  * ✅ SERPAPI using `steamship_langchain.tools.SteamshipSERP`
* Memory
  * ✅ Complete Memory using `steamship_langchain.memory.ConversationBufferMemory`
  * ✅ Windowed Memory using ``steamship_langchain.memory.ConversationBufferWindowMemory``
* ⚒️ Utils 
  * ✅ VectorStores using `steamship_langchain.vectorstore.SteamshipVectorStore`

## Use Cases

* Chatbot 
  * ✅ ChatGPT


