# Quickstart Guide

This tutorial explains how to move your LangChain app into production using Steamship.

## Installation

To get started, install Steamship's LangChain support library with the following command:

```bash
pip install steamship-langchain
```


## Environment Setup

Steamship's backend provides access to external model providers, data stores, and apis without the need to install additional libraries or sign up for new API Keys. 

To get access, you'll need to store your (Steamship API key)[https://steamship.com/account/api.] as an environment variable:

```bash
export STEAMSHIP_API_KEY="..."
```

Alternatively, you pass it directly to the Steamship client:

```python
client = Steamship(api_key="")
```


## Deploying your LangChain app 

Deploying your LangChain app to Steamship involves 3 simple steps: 

1. Use Steamship's adapters 
2. Define your API endpoints 
3. Ship deploy 

### Step 1: Use Steamship's adapters 

Using Steamship's adapters will instruct your LangChain to use our infrastructure. Today we have adapters for LLMs, Memory, and Tools. 

For a detailed overview of our adapters, check out the adapters overview page.

```diff
- from langchain.llms import OpenAI
+ from steamship_langchain.llms import OpenAI

+ client = Steamship()

llm = OpenAI(
+   client=client,
    model_name="text-ada-001", 
    n=2, best_of=2, 
    temperature=0.9)

output = llm("Tell me a joke")
```

### Step 2: Create a Steamship package 

Steamship apps run behind API endoints that are scalable and secure. 
To turn your LangChain into an API you'll have to wrap it in our PackageService scaffolding. 
A PackageService defines API endpoints using annotations:

```python
from steamship_langchain.llms import OpenAI
from steamship.invocable import PackageService, post

class JokeWizard(PackageService):

    @post("generate")
    def generate(self) -> str:
        llm = OpenAI(
            client=self.client,
            model_name="text-ada-001", 
            n=2, best_of=2, 
            temperature=0.9)
        
        return llm("Tell me a joke")
```

### Step 3: Ship deploy 

That's it, now you can deploy your package:

```bash
ship it
```