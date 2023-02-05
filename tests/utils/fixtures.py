import pytest
from steamship import Steamship, Workspace
from utils.client import get_steamship_client


@pytest.fixture()
def client() -> Steamship:
    """Returns a client rooted in a new workspace, then deletes that workspace afterwards.

    To use, simply import this file and then write a test which takes `client`
    as an argument.

    Example
    -------
    The client can be used by injecting a fixture as follows::

        @pytest.mark.usefixtures("client")
        def test_something(client):
          pass
    """
    steamship = get_steamship_client()
    workspace = Workspace.create(client=steamship)
    new_client = get_steamship_client(workspace_id=workspace.id)
    yield new_client
    workspace.delete()
