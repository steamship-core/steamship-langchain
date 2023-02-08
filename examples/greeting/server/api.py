from langchain.prompts.prompt import PromptTemplate
from steamship.invocable import PackageService, post

from steamship_langchain.llms import OpenAI


class GreetingPackage(PackageService):
    @post("greet")
    def greet(self, user: str) -> str:
        prompt = PromptTemplate(
            input_variables=["user"],
            template="Create a welcome message for user {user}. Thank them for running their LangChain app on Steamship. "
            "Encourage them to deploy their app via `ship deploy` when ready.",
        )
        llm = OpenAI(client=self.client, temperature=0.8)
        return llm(prompt.format(user=user))
