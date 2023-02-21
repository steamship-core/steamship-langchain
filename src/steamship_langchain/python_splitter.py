"""Provides a NAIVE splitter for Python code files using the AST.

NOTE: This is meant to provide only a _basic_ chunking for Python source code (for use in embeddings).
It does NOT handle any fine-grained semantic chunking beyond Class + Function chunks. There is no smaller
unit of split than a FunctionDef. As such, if you have (extremely) long functions, you may want to further
split the results from using this class.
"""

import ast
from typing import List, Optional, Union

from langchain.text_splitter import TextSplitter


def _parent_class_lines(python_class, code_lines: List[str]) -> str:
    """Return a string with parent context.

    This is used to ensure individual segments (like nested classes and functions)
    are "annotated" with their parent scope.
    """
    first_line = python_class.first_line
    if first_line > 0:
        first_line -= 1
    if first_line > len(code_lines):
        return ""
    line = code_lines[
        first_line
    ]  # for now, assume the full class declaration lives on a single line
    if python_class.parent:
        return "\n".join([_parent_class_lines(python_class.parent, code_lines), line])
    return line


class PythonSegment:
    """Represents a chunk of Python code in the context of a code file."""

    def __init__(self, name: str, first_line: int = -1, last_line: int = -1):
        """Initialize the segment.

        :param name: the name of the segment
        :param first_line: the index of the first line of a segment in the overall code file
        :param last_line: the index of the last line of a segment in the overall code file
        """
        self._parent = None
        self._name = name
        self._first_line = first_line
        self._last_line = last_line

    @property
    def first_line(self):
        return self._first_line

    @property
    def last_line(self):
        return self._last_line

    @property
    def parent(self):
        return self._parent

    def set_parent(self, parent):
        self._parent = parent

    def code(self, code_lines: List[str]) -> List[str]:
        """Return the code listings for this Segment."""
        # handle switch from 1- to 0-indexed
        code = "\n".join(code_lines[self.first_line - 1 : self.last_line])
        if self.parent:
            return ["\n".join([_parent_class_lines(self.parent, code_lines), code])]
        return [code]


class ClassDef(PythonSegment):
    """Represents a segment of code for python class definition."""

    def __init__(self, name: str, first_line: int = -1, last_line: int = -1, docstring=None):
        """Initialize a class segment.

        :param name: the name of the class segment
        :param first_line: the index of the first line of a class segment in the overall code file
        :param last_line: the index of the last line of a class segment in the overall code file
        :param docstring: the associated docstring for the class.
        """
        super().__init__(name=name, first_line=first_line, last_line=last_line)
        self._parent = None
        self._owned_segments = [[first_line, last_line]]
        self._docstring = docstring

    def record_child_segment(self, start: int, end: int):
        """Records a child segment.

        This informs the ClassDef that it does not need to generate
        a code segment for the portion of code covered by the child
        segment.
        """
        for idx, seg in enumerate(self._owned_segments):
            if seg[0] <= start <= seg[1]:
                del self._owned_segments[idx]
                if start - 1 > seg[0]:
                    self._owned_segments.append([seg[0], start - 1])

                if end + 1 < seg[1]:
                    self._owned_segments.append([end + 1, seg[1]])
                break

    @property
    def owned_segments(self):
        return self._owned_segments

    def code(self, code_lines: List[str]) -> List[str]:
        if len(self._owned_segments) == 0:
            return []

        segs = []
        for seg in self._owned_segments:
            # handle switch from 1- to 0-indexed
            code = "\n".join(code_lines[seg[0] - 1 : seg[1]])
            # ignore empty segments, there is no point in indexing them
            if len(code.strip()) == 0:
                continue
            # always provide context
            if seg[0] != self.first_line:
                segs.append("\n".join([_parent_class_lines(self, code_lines), code]))
            elif self.parent:
                segs.append("\n".join([_parent_class_lines(self._parent, code_lines), code]))
            else:
                segs.append(code)
        return segs


class FuncDef(PythonSegment):
    """Represents a segment of code for python function definition.

    At the moment, this is just an alias for PythonSegment, serving as placeholder for potential future expansion.
    """

    pass


class ClassAndFunctionVisitor(ast.NodeVisitor):
    """Visit Nodes in a Python AST and record important code segments."""

    _segments: List[Union[ClassDef, FuncDef]] = []
    _current_context: Optional[ClassDef] = None

    def __init__(self):
        self._segments = []
        self._current_context = None

    @property
    def segments(self) -> List[Union[ClassDef, FuncDef]]:
        return self._segments

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        """Visit AyncFunctionDef and store segment for extraction.

        NOTE: we don't care about visiting the children as we will assume for now that
        nested functions should not be split from their wrapping contextual function. This
        assumption may need to be re-evaluated at a later time.
        """
        start = node.lineno
        start -= len(node.decorator_list)
        # we don't care about the difference between sync and async for these purposes
        func = FuncDef(name=node.name, first_line=start, last_line=node.end_lineno)

        # update current context
        if self._current_context:
            func.set_parent(self._current_context)
            self._current_context.record_child_segment(start, node.end_lineno)

        # save segment
        self._segments.append(func)

    def visit_FunctionDef(self, node):  # noqa: N802
        """Visit FunctionDef and store segment for extraction.

        NOTE: we don't care about visiting the children as we will assume for now that
        nested functions should not be split from their wrapping contextual function. This
        assumption may need to be re-evaluated at a later time.
        """
        start = node.lineno
        start -= len(node.decorator_list)
        func = FuncDef(name=node.name, first_line=start, last_line=node.end_lineno)

        # update current context
        if self._current_context:
            func.set_parent(self._current_context)
            self._current_context.record_child_segment(start, node.end_lineno)

        # save segment
        self._segments.append(func)

    def visit_ClassDef(self, node):  # noqa: N802
        """Visit ClassDef and store segment for extraction.

        This will build up a PythonClass segment and set it as the current context. This will allow
        proper contextual reporting for nested classes and functions. We allow sub-segments for
        nested class and functions. This can result in disjoint segment generation for class elements.
        """
        cls = ClassDef(node.name, node.lineno, node.end_lineno, docstring=ast.get_docstring(node))

        # update context
        if self._current_context:
            self._current_context.record_child_segment(node.lineno, node.end_lineno)
            cls.set_parent(self._current_context)

        # save segment
        self._segments.append(cls)

        # set current context
        self._current_context = cls
        # visit children
        ast.NodeVisitor.generic_visit(self, node)
        # pop context after visiting children
        self._current_context = self._current_context.parent


class PythonCodeSplitter(TextSplitter):
    """Interpret text as python source and, if beyond a certain line limit, split the text around function
     and class definitions (preserving parent context in splits if possible).

    This represents a **naive** attempt to provide digestible chunks of code for downstream embedding and search.
    It will NOT split up a FunctionDef (even for nested FunctionDefs). It does not attempt to merge disjoint
    ClassDef segments.

    It tries to preserve context for segments (parent classes) where appropriate in an attempt to help
    make the segments more readibly-understandable and relatable.

    Import statements and globals are completely ignored (read: dropped) when chunking is required.
    """

    def __init__(self, max_file_lines: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._max_lines = max_file_lines
        self._visitor = ClassAndFunctionVisitor()

    def split_text(self, text: str) -> List[str]:
        code_lines = text.replace("\r", "\n").split("\n")
        # if the total file size is less than max_lines, just call it a day and move on
        if len(code_lines) <= self._max_lines:
            return [text]
        # get the AST and visit funcdefs and classdefs and chunk
        tree = ast.parse(source=text)
        self._visitor.visit(tree)
        return self._code_segments(code_lines)

    def _code_segments(self, code_lines: List[str]) -> List[str]:
        return [
            snippet.strip() for seg in self._visitor.segments for snippet in seg.code(code_lines)
        ]
