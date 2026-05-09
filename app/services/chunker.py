import re
from dataclasses import dataclass

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int


@dataclass
class _Block:
    text: str
    kind: str  # "prose" | "code" | "table"


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _parse_blocks(text: str) -> list[_Block]:
    lines = text.split("\n")
    blocks: list[_Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("```"):
            start = i
            i += 1
            while i < n and not lines[i].lstrip().startswith("```"):
                i += 1
            if i < n:
                i += 1  # include closing fence
            blocks.append(_Block(text="\n".join(lines[start:i]), kind="code"))
            continue

        if stripped.startswith("|"):
            start = i
            while i < n and lines[i].lstrip().startswith("|"):
                i += 1
            blocks.append(_Block(text="\n".join(lines[start:i]), kind="table"))
            continue

        if line.strip() == "":
            i += 1
            continue

        start = i
        while i < n:
            ll = lines[i].lstrip()
            if lines[i].strip() == "" or ll.startswith("```") or ll.startswith("|"):
                break
            i += 1
        blocks.append(_Block(text="\n".join(lines[start:i]), kind="prose"))

    return blocks


def _split_on_headings(text: str) -> list[str]:
    lines = text.split("\n")
    pieces: list[list[str]] = [[]]
    for line in lines:
        if re.match(r"^#{1,6} ", line) and any(p.strip() for p in pieces[-1]):
            pieces.append([line])
        else:
            pieces[-1].append(line)
    return ["\n".join(p) for p in pieces if any(line.strip() for line in p)]


def _split_sentences(text: str) -> list[str]:
    parts = text.replace(". ", ".\x00").split("\x00")
    return [p.strip() for p in parts if p.strip()]


def _split_prose(text: str, max_tokens: int) -> list[str]:
    if _count_tokens(text) <= max_tokens:
        return [text]
    out: list[str] = []
    for piece in _split_on_headings(text):
        if _count_tokens(piece) <= max_tokens:
            out.append(piece)
            continue
        buf = ""
        for sentence in _split_sentences(piece):
            cand = (buf + " " + sentence).strip() if buf else sentence
            if buf and _count_tokens(cand) > max_tokens:
                out.append(buf)
                buf = sentence
            else:
                buf = cand
        if buf:
            out.append(buf)
    return out


def _take_last_tokens(text: str, n: int) -> str:
    tokens = _ENCODING.encode(text)
    if len(tokens) <= n:
        return text
    return _ENCODING.decode(tokens[-n:])


def chunk_markdown(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    min_tokens: int = 100,
) -> list[Chunk]:
    blocks = _parse_blocks(text)
    if not blocks:
        return []

    # Reserve room for overlap so even chunks that get an overlap prefix
    # stay under max_tokens.
    pack_target = max(1, max_tokens - overlap_tokens)

    expanded: list[_Block] = []
    for b in blocks:
        if b.kind == "prose" and _count_tokens(b.text) > pack_target:
            for piece in _split_prose(b.text, pack_target):
                expanded.append(_Block(text=piece, kind="prose"))
        else:
            expanded.append(b)

    grouped: list[list[_Block]] = []
    current: list[_Block] = []
    current_tokens = 0
    for b in expanded:
        b_tokens = _count_tokens(b.text)
        if not current:
            current = [b]
            current_tokens = b_tokens
            continue
        if current_tokens + b_tokens <= pack_target:
            current.append(b)
            current_tokens += b_tokens
        else:
            grouped.append(current)
            current = [b]
            current_tokens = b_tokens
    if current:
        grouped.append(current)

    metas = [
        ("\n\n".join(b.text for b in g), g[0].kind, g[-1].kind)
        for g in grouped
    ]

    final: list[str] = []
    for i, (chunk_text, first_kind, _last_kind) in enumerate(metas):
        if i == 0:
            final.append(chunk_text)
            continue
        _prev_text, _, prev_last_kind = metas[i - 1]
        if prev_last_kind == "prose" and first_kind == "prose":
            overlap = _take_last_tokens(metas[i - 1][0], overlap_tokens)
            final.append(overlap + " " + chunk_text)
        else:
            final.append(chunk_text)

    if len(final) > 1 and _count_tokens(final[-1]) < min_tokens:
        final[-2] = final[-2] + "\n\n" + final[-1]
        final.pop()

    return [
        Chunk(index=i, text=t, token_count=_count_tokens(t))
        for i, t in enumerate(final)
    ]
