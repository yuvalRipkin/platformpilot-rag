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
    # Each paragraph uses a unique English vocabulary so the overlapping
    # tail is unambiguous — a phrase that appears at the start of chunk
    # N+1 must have come from chunk N's tail, not from elsewhere.
    paragraphs = [
        "The configuration system establishes baseline parameters from "
        "environment files. Each runtime evaluates predicates against "
        "current observations. Deployment manifests describe expected "
        "topology and capacity. Rollouts proceed through staged groups "
        "with health gates between steps.",
        "Telemetry pipelines aggregate signals from edge collectors into "
        "central indexes. Retention policies prune historical samples on "
        "rolling windows. Alert routes branch by severity classes and "
        "ownership labels. Dashboards summarize throughput, error rate, "
        "and saturation per service.",
        "Authentication flows exchange short-lived bearer tokens between "
        "trusted services. Authorization layers enforce policy through "
        "attribute evaluation. Audit logs capture every privileged "
        "decision with structured context. Secrets rotate on schedules "
        "managed by external key brokers.",
        "Indexing strategies balance write amplification against query "
        "latency. Compaction thresholds adjust to the observed mix of "
        "reads and writes. Replicas elect leaders through quorum "
        "protocols. Recovery procedures restore lost segments from "
        "replicated journals when nodes fail.",
        "Capacity planning forecasts saturation before user-visible "
        "degradation occurs. Synthetic probes detect regression early in "
        "release pipelines. Chaos experiments validate failure handling "
        "under controlled disruption. Postmortems document timelines, "
        "root causes, and follow-up actions.",
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunk_markdown(text, max_tokens=120, overlap_tokens=30, min_tokens=10)
    assert len(chunks) >= 2

    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_words = prev.text.split()
        nxt_head = " ".join(nxt.text.split()[:25])
        # Some 3-word window from late in the previous chunk must appear
        # contiguously near the start of the next chunk.
        candidates = prev_words[-15:]
        found = any(
            " ".join(candidates[i : i + 3]) in nxt_head
            for i in range(max(0, len(candidates) - 2))
        )
        assert found, (
            f"no contiguous 3-word overlap between chunk {prev.index} "
            f"and chunk {nxt.index}; prev tail words: {candidates[-6:]}, "
            f"next head: {nxt_head[:120]!r}"
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
