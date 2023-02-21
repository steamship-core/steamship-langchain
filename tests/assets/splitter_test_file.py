"""This is used exclusively to test the python text splitter.

Segment comments are added to highlight where we expect the splitter to
chunk up the code. Care is taken to fold the comments in places that do
not result in extraneous segments.
"""
from typing import TypeVar

DROPPED = TypeVar("DROPPED")
IGNORED = "foo"


# segment
def foo(baz: str) -> str:
    x = 4 * 2
    y = x * 5
    return f"{baz}: {x}, {y}"


# segment
class FooClass:
    """Test class with some docstring.

    This is just an empty, meaningless chunk of text. Sorry to waste your time.
    """

    def __init__(self, **kwargs):
        # segment
        self._nothing = ""
        self._yoyo = "ma"

    class NestedFoo:
        """Nest inside of Foo."""

        # segment
        bar_alias: str
        baz: str
        steamship: bool
        # Some worthless comment.
        # How bout a TODO?
        false = True

        class DoublyNestedFoo:
            def bar(self) -> []:
                # segment
                pass

    @property
    def has_a_decorator(self):
        # segment
        return "ooh, fancy"

    def func_with_nesting(self):
        # segment
        def nested():
            print("we must go deeper")

        return nested()

    foo_help: str  # segment


# segment
class SubFoo(FooClass):
    pass


# segment
async def bar(none: str):
    print("this does nothing and likes it.")
    return
