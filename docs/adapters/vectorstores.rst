VectorStores
============

Steamship's vector store can be accessed using the ``steamship_langchain.vectorstores.SteamshipVectorStore``.
``SteamshipVectorStore`` implements LangChain's VectorStore interface and can be used to replace any vectorstores offered by LangChain.


The following notebooks demonstrates a few of the possible use-cases:

- `Question answering over a vector database <./vectorstores/examples/vector_db_qa.html>`_: This allows be useful for when you have a LOT of documents, and you don’t want to pass them all to the LLM, but rather first want to do some semantic search over embeddings.


.. toctree::
   :maxdepth: 1
   :name: VectorStores
   :hidden:

   ./vectorstores/examples/vector_db_qa.ipynb