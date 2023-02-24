Callbacks
=========


======================
LoggingCallbackHandler
======================

When developing LangChain apps locally, it is often useful to turn on verbose logging
to help debug behavior and performance. The ``LoggingCallbackHandler`` preserves this
capability when deploying your app to Steamship.

To use the ``LoggingCallbackHandler``, set the global callback handler and set ``verbose=True`` in your chain::

    from steamship_langchain.callbacks import LoggingCallbackHandler

    langchain.set_handler(LoggingCallbackHandler())

    ...

    chain = load_qa_with_sources_chain(llm, verbose=True)


Then, to view the logs, retrieve them with ``ship logs``::

    $ ship logs -w my-workspace | jq '.entries[].message'

    "Finished chain."
    "Response data size 11315"
    "Finished chain."
    "Got workspace: my-workspace/..."
    "Entering new LLMChain chain..."
    "Prompt after formatting:\n\u001b[32;1m\u001b[1;3mGiven the following extracted parts ...
    "Entering new StuffDocumentsChain chain..."

