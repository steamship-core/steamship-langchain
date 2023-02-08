"""Demonstration of calling OpenAI's LLM without Steamship"""
from steamship import Steamship

from steamship_langchain import OpenAI

client = Steamship()

llm = OpenAI(client=client, model_name="text-ada-001", n=2, best_of=2, temperature=0.9)
# TODO: Stop using client=client, instead use steamship_langchain.client = client

completion = llm("Tell me a joke")
print(completion)

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 15)
len(llm_result.generations)
print(llm_result.generations[0])

print(llm_result.llm_output)

print(llm.get_num_tokens("what a joke"))

# [Generation(text='\n\nWhy did the chicken cross the road?\n\nTo get to the other side!', generation_info={'finish_reason': 'stop', 'logprobs': None}), Generation(text='\n\nWhy did the chicken cross the road?\n\nTo get to the other side.', generation_info={'finish_reason': 'stop', 'logprobs': None})]
# {'token_usage': {'total_tokens': 3673, 'completion_tokens': 3553, 'prompt_tokens': 120}}
# 3
