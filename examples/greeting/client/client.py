from steamship import RuntimeEnvironments, Steamship, check_environment
from termcolor import colored


def main():
    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.LOCALHOST)

    with Steamship.temporary_workspace() as client:
        # This handle MUST match the handle that you deployed with. Here we use the default option.
        api = client.use(package_handle="test-simple-greeting")

        while True:
            name = input(colored("Name: ", "blue"))
            print(colored(f'{api.invoke("/greet", user=name).strip()}\n', "green"))


if __name__ == "__main__":
    main()
