# QA With Sources Example (Embeddings)

This is an example of deploying an app that uses a `qa_with_sources_chain` LangChain on Steamship.
Steamship's embeddings support is used to find and provide sources for the inference.

## Try it out

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Question-Answering-with-Sources-using-LangChain-on-Steamship)](https://replit.com/@SteamshipDoug/Question-Answering-with-Sources-using-LangChain-on-Steamship)

Or, install dependencies:
```commandline
pip install steamship-langchain
pip install termcolor
```

And run the client:
```commandline
python client/client.py
```

## Deploy your own

Switch to the `server/` directory and run deploy.
```commandline
cd server
ship deploy
```

The deployment script will walk you through setting up a package name that you 
can use for your own instance, if desired. This will enable you to modify the example
to meet your needs (or just to have fun experimenting with).

After deployment, switch back to the parent directory (`$ cd ..`) to run the client, etc.
You'll need to update the `package_handle` in the client to match your new deployment.

### api.py

`api.py` is required by Steamship for packages being deployed. You may add additional source
files as desired, but you MUST always have `api.py`.

### A note on dependencies

Steamship relies on `requirements.txt` as part of the packaging and deploy. If you add
new dependencies to your server code, please ensure they are reflected in `requirements.txt`.