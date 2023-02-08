import pytest
from steamship import Steamship

from steamship_langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

TEST_PROMPT = "this is a test: "
LLM_STRING = "llm"
UNKNOWN = "unknown"


@pytest.mark.usefixtures("client")
def test_persistent_memory(client: Steamship):
    # example responses heavily borrowed from:
    # https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html
    memory_under_test = ConversationBufferMemory(client=client, key="user-1234-session-1")

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert memory_variables["history"] == ""

    inputs = {"history": "", "human_input": "ls ~"}
    outputs = {
        "response": """
```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```
"""
    }
    memory_under_test.save_context(inputs, outputs)

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert (
        memory_variables["history"]
        == """Human: ls ~
AI: \n```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```
"""
    )

    inputs = {"history": "ignored", "human_input": "pwd"}
    outputs = {
        "response": """
```
$ pwd
/
```"""
    }
    memory_under_test.save_context(inputs, outputs)

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert (
        memory_variables["history"]
        == """Human: ls ~
AI: \n```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```

Human: pwd
AI: \n```
$ pwd
/
```"""
    )

    inputs = {"history": "ignored", "human_input": "ping bbc.com"}
    outputs = {
        "response": """
```
$ ping bbc.com
PING bbc.com (151.101.65.81): 56 data bytes
64 bytes from 151.101.65.81: icmp_seq=0 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=1 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=2 ttl=53 time=14.945 ms

--- bbc.com ping statistics ---
3 packets transmitted, 3 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 14.945/14.945/14.945/0.000 ms
```
"""
    }
    memory_under_test.save_context(inputs, outputs)

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert (
        memory_variables["history"]
        == """Human: ls ~
AI: \n```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```

Human: pwd
AI: \n```
$ pwd
/
```
Human: ping bbc.com
AI: \n```
$ ping bbc.com
PING bbc.com (151.101.65.81): 56 data bytes
64 bytes from 151.101.65.81: icmp_seq=0 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=1 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=2 ttl=53 time=14.945 ms

--- bbc.com ping statistics ---
3 packets transmitted, 3 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 14.945/14.945/14.945/0.000 ms
```
"""
    )


@pytest.mark.usefixtures("client")
def test_persistent_window_memory(client: Steamship):
    # example responses heavily borrowed from:
    # https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html
    memory_under_test = ConversationBufferWindowMemory(
        client=client, key="user-1234-session-2", k=2
    )

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert memory_variables["history"] == ""

    inputs = {"history": "", "human_input": "ls ~"}
    outputs = {
        "response": """
```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```
"""
    }
    memory_under_test.save_context(inputs, outputs)

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert (
        memory_variables["history"]
        == """Human: ls ~
AI: \n```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```
"""
    )

    inputs = {"history": "ignored", "human_input": "pwd"}
    outputs = {
        "response": """
```
$ pwd
/
```"""
    }
    memory_under_test.save_context(inputs, outputs)

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert (
        memory_variables["history"]
        == """Human: ls ~
AI: \n```
$ ls ~
Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos
```

Human: pwd
AI: \n```
$ pwd
/
```"""
    )

    inputs = {"history": "ignored", "human_input": "ping bbc.com"}
    outputs = {
        "response": """
```
$ ping bbc.com
PING bbc.com (151.101.65.81): 56 data bytes
64 bytes from 151.101.65.81: icmp_seq=0 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=1 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=2 ttl=53 time=14.945 ms

--- bbc.com ping statistics ---
3 packets transmitted, 3 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 14.945/14.945/14.945/0.000 ms
```
"""
    }
    memory_under_test.save_context(inputs, outputs)

    memory_variables = memory_under_test.load_memory_variables(inputs={})
    assert len(memory_variables) == 1
    assert memory_variables["history"] is not None
    assert (
        memory_variables["history"]
        == """Human: pwd
AI: \n```
$ pwd
/
```
Human: ping bbc.com
AI: \n```
$ ping bbc.com
PING bbc.com (151.101.65.81): 56 data bytes
64 bytes from 151.101.65.81: icmp_seq=0 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=1 ttl=53 time=14.945 ms
64 bytes from 151.101.65.81: icmp_seq=2 ttl=53 time=14.945 ms

--- bbc.com ping statistics ---
3 packets transmitted, 3 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 14.945/14.945/14.945/0.000 ms
```
"""
    )
