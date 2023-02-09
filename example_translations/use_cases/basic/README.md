# Basic Concepts 

This example demonstrates how to use Steamship compatible LLMs in LangChain app. 
Steamship compatible LLMs respect the LangChain's standard LLM interface and can easily be swapped by updating the import statement.
Doing so will give you access to our distributed that scale with the number of requests. 
Since these LLMs are run in Steamship's cloud infrastructure you won't have to worry about installing upstream dependencies to run these LLMs. 

All our LLMs come preloaded with Api Keys. For more information about API Key management and billing, please read our access & billing section. 

## Supported LLM Providers

Steamship supports 1 LLM provider. Support for more LLM providers is on its way. 

| Provider    | LangChain            | Steamship                       | 
|-------------|----------------------|---------------------------------|
| OpenAI      | langchain.llms.OpenAI | steamship_langchain.llms.OpenAI |

## Example 

For this example, we will work with an OpenAI LLM wrapper, although the functionalities highlighted are generic for all LLM types.

```diff
- from langchain.llms import OpenAI
+ from steamship_langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", n=2, best_of=2, temperature=0.9)

print("Completion:", llm("Tell me a joke"))

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)

print("n_completions", len(llm_result.generations)) 

print("first completion", llm_result.generations[0]) 

print("last completion", llm_result.generations[-1]) 

print("llm_output", llm_result.llm_output)

print("number of tokens", llm.get_num_tokens("what a joke")) 

```



