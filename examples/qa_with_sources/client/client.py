from pathlib import Path

from steamship import Block, File, RuntimeEnvironments, Steamship, check_environment
from termcolor import colored

STATE_OF_THE_UNION_PATH = (
    Path(__file__).parent.parent.parent.parent / "docs" / "state_of_the_union.txt"
)


def main():
    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.LOCALHOST)
    exit()

    with Steamship.temporary_workspace() as client:
        # This handle MUST match the handle that you deployed with. Here we use the default option.
        api = client.use(package_handle="test-qa-with-sources")

        # Embed the State of the Union address
        with STATE_OF_THE_UNION_PATH.open() as f:
            print(
                colored("Saving the state of the union file to Steamship workspace...", "blue"),
                end="",
                flush=True,
            )
            sotu_file = File.create(client, blocks=[Block(text=f.read())])
            print(colored("Done.", "blue"))

        print(colored("Indexing state of the union...", "blue"), end="", flush=True)
        api.invoke("/index_file", file_handle=sotu_file.handle)
        print(colored("Done.", "blue"))

        # Issue Query
        query = "What did the president say about Justice Breyer?"
        print(colored("\nQuery: ", "blue"), f"{query}")

        print(colored("Awaiting results. Please be patient. This may take a few moments.", "blue"))

        response = api.invoke("/qa_with_sources", query=query)
        print(colored("Answer: ", "blue"), f"{response['result'].strip()}")

        # Print sources (with text)
        sources = response["source_documents"]

        if not sources:
            print(colored("No sources provided in response.", "red"))
            return

        for source in sources:
            print(source.get("page_content", "Source text missing"))


if __name__ == "__main__":
    main()
