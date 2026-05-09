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
    text = (
        "The quick brown fox jumps over the lazy dog in the morning. " * 30
        + "\n\n"
        + "Pack my box with five dozen liquor jugs at the market today. " * 30
    )
    chunks = chunk_markdown(text, max_tokens=120, overlap_tokens=20, min_tokens=10)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_tail_words = prev.text.split()[-10:]
        next_start = " ".join(nxt.text.split()[:30])
        found = any(
            " ".join(prev_tail_words[i : i + 3]) in next_start
            for i in range(len(prev_tail_words) - 2)
        )
        assert found, f"no real overlap chunk {prev.index}→{nxt.index}"


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
