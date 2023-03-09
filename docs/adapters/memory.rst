Memory
======

By default, LangChain chains and agents are stateless. To preserve state across executions,
LangChain supports the concept of Memory. Memory is particulary useful for building chat-bots, for example.

Steamship was built with stateful applications in mind. We offer stateful storage of chat
history for use with LangChain to preserve contextual history across user sessions, etc. in
production environments:

- ``steamship_langchain.memory.ChatMessageHistory``

Please see the following notebook for a demonstration of Steamship's persistent memory in action:

- `How-To Guide <./memory/how_to_guide.html>`_: This guide shows how to use the ``steamship_langchain.memory`` objects to save state.



.. toctree::
   :maxdepth: 1
   :caption: Memory
   :name: Memory
   :hidden:

   ./memory/how_to_guide.rst