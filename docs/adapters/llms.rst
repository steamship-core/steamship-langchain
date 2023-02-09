LLMs
==========================

Large Language Models (LLMs) are supported by Steamship using Steamship compatible LLMs found under ``steamship_langchain.llms``.
Using Steamship LLM adapters gives you access to our distributed cloud, using our cloud vs local LLMs has several advantages:

* ✅ You won't have to worry about installing upstream dependencies.
* ✅ You won't have to set up billing to get vendor-specific API KEY
* ✅ You're LLM invocations will scale even when you go viral




The following sections of documentation are provided:

- `Getting Started <./llms/getting_started.html>`_: An overview of how to use Steamship LLM adapters.

- `LLM Caching <./llms/llm_caching.html>`_: Save money by caching results of your individual LLM calls.

- `Reference <../reference/modules/llms.html>`_: API reference documentation for all LLM classes.


.. toctree::
   :maxdepth: 1
   :name: LLMs
   :hidden:

   ./llms/getting_started.ipynb
   ./llms/llm_caching.ipynb
   Reference<../reference/modules/llms.rst>