"""Demonstration of calling OpenAI's LLM without Steamship"""
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", n=2, best_of=2, temperature=0.9)

completion = llm("Tell me a joke")
print(completion)

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 15)
len(llm_result.generations)
print(llm_result.generations[0])

print(llm_result.llm_output)

print(llm.get_num_tokens("what a joke"))
