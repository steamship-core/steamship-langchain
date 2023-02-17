from steamship_langchain.python_splitter import PythonCodeSplitter
from tests import TEST_ASSETS_PATH

TEST_FILE = "splitter_test_file.py"


def test_python_splitter():
    # set up a splitter with a low line number limit to force the test
    test_splitter = PythonCodeSplitter(max_file_lines=5)

    with open(TEST_ASSETS_PATH / TEST_FILE) as f:
        segments = test_splitter.split_text(f.read())

    # There should be 10 segments in the generated set. These are labeled in the source.
    assert len(segments) == 10

    assert segments[0].startswith("def foo(baz: str) -> str:")
    assert segments[1].startswith("class FooClass:")
    assert segments[2].startswith("class FooClass:")  # the help attribute segment
    assert segments[3].startswith("class FooClass:")
    assert segments[4].startswith(
        """class FooClass:
    class NestedFoo:"""
    ), print(segments)
    assert segments[5].startswith(
        """class FooClass:
    class NestedFoo:
        class DoublyNestedFoo:
            def bar(self) -> []:"""
    )
    assert segments[6].startswith(
        """class FooClass:
    @property
    def has_a_decorator(self):"""
    )
    assert segments[7].startswith(
        """class FooClass:
    def func_with_nesting(self):"""
    )
    assert segments[8].startswith("class SubFoo(FooClass):")
    assert segments[9].startswith("async def bar(none: str):")

    long_test_splitter = PythonCodeSplitter(max_file_lines=500)
    with open(TEST_ASSETS_PATH / TEST_FILE) as f:
        full_text = f.read()
        long_segments = long_test_splitter.split_text(full_text)

    assert len(long_segments) == 1
    assert long_segments[0] == full_text
