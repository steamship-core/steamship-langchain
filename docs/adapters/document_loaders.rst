Document Loaders
================

LangChain documents provide a wrapper around a chunk of text, with support for associated metadata. They are
used to combine text from external sources with LLMs in prompts.

Steamship provides a way to create LangChain documents from existing Steamship Files. This export enables
use of cloud-based persistent storage with local chains and more.

The following notebook provides a demonstration of how to use the SteamshipLoader:

- `How-To Guide <./document_loaders/how_to_guide.html>`_: Loading Steamship Files into LangChain Documents.


.. toctree::
   :maxdepth: 1
   :caption: Document Loaders
   :name: Document Loaders
   :hidden:

   ./document_loaders/how_to_guide.ipynb
