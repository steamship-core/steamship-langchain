from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from steamship import Steamship

from steamship_langchain.chat_models.openai import ChatOpenAI

template = """
Who you are:
You are the assistant marketing manger.

What you do:
You read podcast transcripts and extract insights from the podcast in response to a question.
You extract only big, key information pieces. Ignore mundane information. We have limited memory.

Be descriptive with the insights you find. Return value in a descriptive, written sentence (if you can find anything).
Make sure to quote the original source and avoid reciting.

Okay, let's go!

"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = """Please extract insights from the podcast transcript to answer this question:{question}

podcst transcript: {podcast_transcript}"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
client = Steamship(workspace="test")
MODEL_NAME = "gpt-3.5-turbo"

topic = "funny stories"
question = f"What unique tips & tricks are being shared about {topic}?"
transcript = """funny stories are meant to be ignored, life sucks as it is. Take it or leave it."""

chat_prompt.format_prompt(question=question, podcast_transcript=transcript[:10_000]).to_messages()

messages = chat_prompt.format_prompt(question=question, podcast_transcript=transcript[:10_000]).to_messages()

# Pure langchain
#
# from langchain.chat_models import ChatOpenAI
#
# chat = ChatOpenAI(
#         model_name=MODEL_NAME,
#         temperature=0,
#         openai_api_key="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",
#     )

# response = chat(chat_prompt.format_prompt(question=question, podcast_transcript=transcript[:10_000]).to_messages())
# response.content


chat = ChatOpenAI(
    client=client,
    model_name=MODEL_NAME,
    temperature=0,  # no hallucinations
)

topic = "funny stories"
question = f"What unique tips & tricks are being shared about {topic}?"
transcript = """funny stories are meant to be ignored, life sucks as it is. Take it or leave it."""

chat_prompt.format_prompt(question=question, podcast_transcript=transcript[:10_000]).to_messages()

messages = chat_prompt.format_prompt(question=question, podcast_transcript=transcript[:10_000]).to_messages()

response = chat(chat_prompt.format_prompt(question=question, podcast_transcript=transcript[:10_000]).to_messages())
print(response.content)
