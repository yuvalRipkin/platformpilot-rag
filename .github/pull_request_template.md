## Summary

<!--
What changed and why. One bullet per component: **Name** (`path/to/file.py`),
then the reasoning. Prefer explaining the decision over narrating the diff —
a reviewer can read the diff, they cannot read why you rejected the
alternative.
-->

-

## Test plan

<!--
Only tick what you actually ran, and paste the real output — "29 passed, 10
deselected", not "tests pass". Leave a box unchecked and say why instead of
ticking it optimistically; an honest unchecked box is more useful than a
checked one you are not sure about.
-->

- [ ] `uv run pytest` —
- [ ] `uv run ruff check .` && `uv run ruff format --check .`

## Measured results

<!--
Every claim about speed, size, memory, latency, throughput or resource use
belongs here, with the number behind it, the environment it was measured in,
and the command to reproduce it. Measure in an environment that resembles
where the code runs — laptop numbers for a containerised service are not
evidence.

If this change makes no such claim, replace the table with "No performance
claim." and move on.

Words like "faster", "smaller", "lighter" or "should improve" anywhere in this
PR without a row here are the thing this section exists to catch.
-->

| metric | before | after | how it was measured |
|---|---|---|---|
|  |  |  |  |

## Notes

<!--
Decisions a reviewer would otherwise have to ask about: deviations from the
brief, tradeoffs knowingly accepted, regressions left in on purpose, anything
that looks wrong but is deliberate. Delete this section if there is nothing.
-->

## Out of scope (follow-ups)

<!-- Deliberately not done here, so review does not re-litigate it. Delete if empty. -->
