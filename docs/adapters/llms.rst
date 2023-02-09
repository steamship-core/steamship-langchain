LLMs
==========================

Large Language Models (LLMs) are supported by Steamship using Steamship compatible LLMs found under `steamship_langchain.llms`.
Using Steamship LLM adapters gives you access to our distributed cloud, using our cloud vs local LLMs has several advantages:

* ✅ You won't have to worry about installing upstream dependencies.
* ✅ You won't have to set up billing to get vendor-specific API KEY
* ✅ You're LLM invocations will scale even when you go viral




The following sections of documentation are provided:

- `Getting Started <./llms/getting_started.html>`_: An overview of how to use Steamship LLM adapters.

- `How-To Guides <./llms/llm_caching.html>`_: A collection of how-to guides. These highlight how to accomplish various objectives with our LLM class, as well as how to integrate with various LLM providers.

- `Reference <../reference/modules/llms.html>`_: API reference documentation for all LLM classes.


.. toctree::
   :maxdepth: 1
   :name: LLMs
   :hidden:

   ./llms/getting_started.ipynb
   ./llms/llm_caching.ipynb
   Reference<../reference/modules/llms.rst>