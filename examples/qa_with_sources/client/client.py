from steamship import Block, File, RuntimeEnvironments, Steamship, check_environment
from termcolor import colored


def main():
    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.LOCALHOST)

    with Steamship.temporary_workspace() as client:
        # This handle MUST match the handle that you deployed with. Here we use the default option.
        api = client.use(package_handle="test-qa-with-sources")

        # Embed the State of the Union address
        with open("state_of_the_union.txt") as f:
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
        print(colored("Answer: ", "blue"), f"{response['output_text'].strip()}")

        # Print sources (with text)
        last_line = response["output_text"].splitlines()[-1:][0]

        if "SOURCES: " not in last_line:
            print(last_line)
            print(colored("No sources provided in response.", "red"))
            return

        sources_list = last_line[len("SOURCES: ") :]

        for source in sources_list.split(","):
            print(colored(f"\nSource text ({source.strip()}):", "blue"))
            for input_doc in response["input_documents"]:
                metadata = input_doc.get("metadata", {})
                src = metadata["source"]
                if source.strip() == src:
                    print(input_doc.get("page_content", "Source text missing"))


if __name__ == "__main__":
    main()
