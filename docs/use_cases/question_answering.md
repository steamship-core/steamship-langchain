# Question Answering

> :warning: **In memory vector stores (e.g. FAISS) are only suitable for stateless invocation. 
> When you want to use a stateful database it is recommended to use the `SteamshipVectorStore` **


Steamship supports QA across databases using all four types of chains: `stuff`, `map_reduce`, `refine`, and `map-rerank`.

For a more in depth explanation of what these chain types are, see [here](https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs.html).

We recommend you use `SteamshipVectorStore` for querying across a large corpus.