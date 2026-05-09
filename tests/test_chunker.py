from app.services.chunker import _count_tokens, chunk_markdown


def test_simple_paragraphs():
    text = (
        "This is the first paragraph with a few sentences in it.\n\n"
        "This is the second paragraph, also short.\n\n"
        "And a third short paragraph to round things out."
    )
    chunks = chunk_markdown(text, max_tokens=500, overlap_tokens=50, min_tokens=10)
    assert len(chunks) == 1
    assert "first paragraph" in chunks[0].text
    assert "third short" in chunks[0].text


def test_long_prose_splits():
    paragraph = (
        "This is a sentence that contains some content. " * 8
    )
    text = "\n\n".join([paragraph] * 20)
    max_tokens = 200
    chunks = chunk_markdown(
        text, max_tokens=max_tokens, overlap_tokens=20, min_tokens=10
    )
    assert len(chunks) > 1
    for c in chunks:
        assert _count_tokens(c.text) <= max_tokens


def test_overlap_present():
    paragraph = (
        "Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango. "
    )
    text = "\n\n".join([paragraph * 4] * 6)
    chunks = chunk_markdown(
        text, max_tokens=120, overlap_tokens=20, min_tokens=10
    )
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_tail = prev.text[-80:]
        next_head = nxt.text[:80]
        # Some non-trivial substring of the previous chunk's tail must appear
        # at the start of the next chunk.
        overlap_words = [w for w in prev_tail.split() if len(w) >= 4]
        assert any(w in next_head for w in overlap_words[-5:]), (
            f"no overlap between chunk {prev.index} and {nxt.index}"
        )


def test_code_block_kept_whole():
    code_lines = ["    line " + str(i) + " of code in this block" for i in range(120)]
    code_block = "```python\n" + "\n".join(code_lines) + "\n```"
    text = "Some intro prose paragraph.\n\n" + code_block + "\n\nClosing prose."
    code_tokens = _count_tokens(code_block)
    assert code_tokens > 500  # set up the precondition

    chunks = chunk_markdown(
        text, max_tokens=500, overlap_tokens=50, min_tokens=10
    )
    matches = [c for c in chunks if "line 0 of code" in c.text]
    assert len(matches) == 1
    assert "line 119 of code" in matches[0].text


def test_heading_boundary_preferred():
    section = "Some sentence about a topic. " * 30
    text = (
        "# Section A\n\n" + section
        + "\n\n## Section B\n\n" + section
        + "\n\n### Section C\n\n" + section
    )
    chunks = chunk_markdown(text, max_tokens=200, overlap_tokens=20, min_tokens=10)
    # Each heading should appear at the start of some chunk (after potential
    # overlap), demonstrating that heading boundaries broke the splits.
    starts_with_heading = [c for c in chunks if "# Section" in c.text[:200]]
    assert len(starts_with_heading) >= 2


def test_min_tokens_merge():
    body = "This is a normal paragraph with several sentences in it. " * 15
    tail = "Tiny trailing line."
    text = body + "\n\n" + tail
    chunks = chunk_markdown(text, max_tokens=200, overlap_tokens=10, min_tokens=50)
    assert tail in chunks[-1].text
    # Verify the tail wasn't kept as its own under-min chunk.
    if len(chunks) > 1:
        assert _count_tokens(chunks[-1].text) >= 50
