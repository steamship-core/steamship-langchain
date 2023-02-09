Welcome to Steamship 4 LangChain
==========================

⚡ Deploying 🦜️🔗LangChain LLM apps to production ⚡


Getting Started
----------------

Checkout the below guide for a walkthrough of how to get started moving Language Model application to production.

- `Getting Started Documentation <./getting_started/getting_started.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting_started
   :hidden:

   getting_started/getting_started.md


Adapters
-----------

To leverage the module


are six main modules that LangChain provides support for.
For each module we provide some examples to get started, how-to guides, reference docs, and conceptual guides.
These modules are, in increasing order of complexity:


- `Prompts <./adapters/prompts.html>`_: This includes prompt management, prompt optimization, and prompt serialization.

- `LLMs <./adapters/llms.html>`_: This includes a generic interface for all LLMs, and common utilities for working with LLMs.

- `Chains <./adapters/chains.html>`_: Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

- `Agents <./adapters/agents.html>`_: Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end to end agents.

- `Memory <./adapters/memory.html>`_: Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.


.. toctree::
   :maxdepth: 1
   :caption: Adapters
   :name: adapters
   :hidden:

   ./adapters/prompts.md
   ./adapters/llms.md
   ./adapters/chains.md
   ./adapters/agents.md
   ./adapters/memory.md

Use Cases
----------

The above adapters cover most of LangChain's use-cases. Below are some of the common use cases supported by Steamship.

- `Agents <./use_cases/chatgpt.html>`_: Agents are systems that use a language model to interact with other tools. These can be used to do more grounded question/answering, interact with APIs, or even take actions.


.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   ./use_cases/chatgpt.md


Coverage and Roadmap
-----------

Support for LangChain is currently in alpha. But we're all in on creating the best LangChain infra stack to date.

We've compiled a status page to track what modules & use-cases are officially supported by Steamship.

Something's missing? We'd love to hear from you. Please reach out to: hello@steamship.com, or join us on our Discord.


- `Our Coverage <./getting_started/getting_started.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Coverage
   :name: coverage
   :hidden:

   support/coverage.md


Feedback and Support
-----------
Have any feedback on this package? Or on Steamship in general?

We'd love to hear from you. Please reach out to: hello@steamship.com, or join us on our Discord.