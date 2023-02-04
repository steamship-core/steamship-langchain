import pytest
from steamship import Steamship

from steamship_langchain.tools import SteamshipSERP


@pytest.mark.usefixtures("client")
def test_search_tool(client: Steamship):
    tool_under_test = SteamshipSERP(client=client)

    answer = tool_under_test.search("Who won the 2019 World Series?")
    assert len(answer) != 0
    assert answer == "Washington Nationals"
