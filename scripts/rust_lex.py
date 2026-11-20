# SPDX-License-Identifier: MIT OR Apache-2.0
"""A minimal Rust lexer for the source lints (check_py_entries.py,
check_py_callouts.py): comments removed, string and character literals
emptied, so neither a `//` inside a string nor a brace or a keyword inside a
string or a comment changes what a lint sees. Line breaks are kept.

It is a lint helper, not a parser: it knows comments (nested block
comments included), string literals (`"..."`, `b"..."`, `c"..."`, raw
`r#"..."#` in every prefix), character and byte literals, and tells a
character literal from a lifetime.
"""

import re

_RAW = re.compile(r'(?:b|c)?r(#*)"')
_PREFIXED = re.compile(r'(?:b|c)"')
_CHAR = re.compile(r"(?:b)?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'")


def strip(text: str) -> str:
    """`text` without comments and with every string and character literal
    emptied (`""`, `' '`); everything else, line breaks included, kept."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    if text[i] == "\n":
                        out.append("\n")
                    i += 1
            out.append(" ")
            continue
        word_start = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        m = _RAW.match(text, i) if word_start else None
        if m:
            end = text.find('"' + m.group(1), m.end())
            end = n if end < 0 else end + 1 + len(m.group(1))
            out.append('""' + "\n" * text.count("\n", i, end))
            i = end
            continue
        m = _PREFIXED.match(text, i) if word_start else None
        if c == '"' or m:
            j = m.end() if m else i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append('""' + "\n" * text.count("\n", i, j))
            i = j + 1
            continue
        m = _CHAR.match(text, i) if c in "'b" and (c == "'" or word_start) else None
        if m:
            out.append("' '")
            i = m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


def block_end(code: str, open_brace: int) -> int:
    """The index just past the `}` matching the `{` at `open_brace`, in
    stripped code (`len(code)` if it is unbalanced)."""
    depth, i = 0, open_brace
    while i < len(code):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(code)


def item_brace(code: str, start: int) -> int:
    """The index of the `{` that opens the item whose header starts at
    `start`, in stripped code: the first `{` outside brackets, parentheses,
    braces and generics. -1 if a `;` at depth 0 ends the item first.

    A `<` may be an operator, in a const expression (`{ 1 < 2 }`, `[u8; 1 <<
    3]`): a `>` closes only a `<` on top of the stack, and a `)`, `]` or `}`
    closes everything opened after its own opener, so such a `<` is dropped
    there and never hides the body."""
    stack = []  # the open delimiters: "(", "[", "{" and "<"
    closes = {")": "(", "]": "[", "}": "{"}
    i = start
    while i < len(code):
        ch = code[i]
        if ch == "{" and not stack:
            return i
        if ch in "([{<":
            stack.append(ch)
        elif ch == ">":
            # `->` is not a closing generic.
            if stack and stack[-1] == "<" and code[i - 1] != "-":
                stack.pop()
        elif ch in closes:
            while stack and stack.pop() != closes[ch]:
                pass
        elif ch == ";" and not stack:
            return -1
        i += 1
    return -1
