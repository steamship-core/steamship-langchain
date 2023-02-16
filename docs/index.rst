Welcome to Steamship 4 LangChain
=================================

⚡ Deploying 🦜️🔗LangChain LLM apps to production ⚡

Getting Started
----------------

Checkout the below guide for a walkthrough of how to move your Language Model application to production.

- `Getting Started Documentation <./getting_started/getting_started.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting_started
   :hidden:

   getting_started/getting_started.md


Adapters
-----------

We've built adapters for LangChain's most important modules and components.
Using these adapters enables you to leverage Steamship's infrastructure including its scalable execution framework, caching, and databases.

To guide you through the upgrade process we've included how-to guides and some examples for each LangChain module.

Note: We're in alpha, so we don't have full support on all module's yet. We're confident we cover the most interesting use-cases though.
Have a look at our `Coverage Page <./getting_started/getting_started.html>`_ for an overview of what is supported.


- `Prompts <./adapters/prompts.html>`_

- `LLMs <./adapters/llms.html>`_

- `Chains <./adapters/chains.html>`_

- `VectorStores <./adapters/vectorstores.html>`_

- `Agents <./adapters/agents.html>`_

- `Memory <./adapters/memory.html>`_


.. toctree::
   :maxdepth: 1
   :caption: Adapters
   :name: adapters
   :hidden:

   ./adapters/prompts.md
   ./adapters/llms.rst
   ./adapters/chains.md
   ./adapters/agents.md
   ./adapters/memory.md
   ./adapters/vectorstores.rst

Use Cases
----------

The above adapters cover a ton of LangChain's use-cases. Below are some of the common use cases supported by Steamship.

- `ChatGPT <./use_cases/chatgpt.html>`_


.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   ./use_cases/chatgpt.ipynb


Coverage and Roadmap
----------------------

Support for LangChain is currently in alpha. But we're all in on creating the best LangChain infra stack.

We've compiled a status page to track what modules & use-cases are officially supported by Steamship.

Something's missing? We'd love to hear from you. Please reach out to: hello@steamship.com, or join us on our Discord.


- `Our Coverage <./getting_started/getting_started.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Coverage
   :name: coverage
   :hidden:

   support/coverage.md