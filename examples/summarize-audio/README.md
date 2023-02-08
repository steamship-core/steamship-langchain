# Summarize Audio (Async Chaining)

> [![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#experimental)
>
> Audio transcription support not yet considered fully-production ready on Steamship. We are working hard on
> productionizing support for audio transcription at scale, but there may be some existing issues that you encounter
> as you try this out.

This is an example of deploying an app LangChain for summarization on Steamship.
It provides an integration with one of Steamship's audio transcription plugins
and a small introduction to the Task system.

## Try it out

[![Run on Repl.it](https://replit.com/badge/github/@SteamshipDoug/Summarize-Audio-with-LangChain-and-Steamship)](https://replit.com/@SteamshipDoug/Summarize-Audio-with-LangChain-and-Steamship)

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