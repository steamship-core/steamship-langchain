from steamship import RuntimeEnvironments, Steamship, check_environment
from steamship.utils.url import Verb
from termcolor import colored


def main():
    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.LOCALHOST)

    # NOTE: we use a temporary workspace here as an example.
    # To persist chat history across sessions, etc., use a persistent workspace.
    with Steamship.temporary_workspace() as client:
        # To use a custom instance, you will need to update this to reflect your
        # deployed package handle. If you deploy instances with differing configuration
        # to the same workspace, you may wish to further provide unique instance handles.
        api = client.use(package_handle="test-chatbot")

        # allows for per-user history saving, etc. Here we use an example user ID.
        session_handle = "example-chat-user-0001"
        print(
            colored(
                "Beginning chat session (type 'history' at any time to see a dump of the full conversation "
                "history)...",
                "blue",
            )
        )
        while True:
            msg = input(colored("You: ", "blue"))
            if msg.lower() == "history" or msg.lower() == "transcript":
                print(
                    colored("Transcript:\n", "blue"),
                    f'{api.invoke("/transcript", verb=Verb.GET, chat_history_handle=session_handle)}\n',
                )
            else:
                print(
                    colored("AI: ", "blue"),
                    colored(
                        f'{api.invoke("/send_message", message=msg, chat_history_handle=session_handle).strip()}\n',
                        "green",
                    ),
                )


if __name__ == "__main__":
    main()
