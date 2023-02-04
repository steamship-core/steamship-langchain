from steamship import Configuration, Steamship

TESTING_PROFILE = "test"


def get_steamship_client(fail_if_workspace_exists=False, **kwargs) -> Steamship:
    # Always use the `test` profile
    kwargs["profile"] = TESTING_PROFILE
    return Steamship(
        fail_if_workspace_exists=fail_if_workspace_exists, config=Configuration.parse_obj(kwargs)
    )
