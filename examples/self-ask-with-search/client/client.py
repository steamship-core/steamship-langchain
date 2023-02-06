from steamship import RuntimeEnvironments, Steamship, check_environment
from termcolor import colored


def main():
    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.LOCALHOST)

    with Steamship.temporary_workspace() as client:
        # This handle MUST match the handle that you deployed with. Here we use the default option.
        api = client.use("test-self-ask-with-search")

        while True:
            query = input(colored("Query: ", "grey"))
            print(colored("Please be patient. Generating...", "grey"), flush=True)
            response = api.invoke("/self_ask_with_search", query=query)
            print(colored("Answer: ", "grey"), colored(f'{response["output"].strip()}', "green"))
            print()
            print(colored("Intermediate Steps: ", "grey"))
            step = 1
            for action in response["intermediate_steps"]:
                print(
                    colored(f"{step}. Tool({action[0].tool}): {action[0].tool_input}", "grey"),
                    colored(f"{action[1]}", "green"),
                )
                step = step + 1
            print("\n-----\n")


if __name__ == "__main__":
    main()
