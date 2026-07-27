# Code Style

- Trust architectural invariants. Do not add fallback shapes, backends, or hypothetical-case defenses in kernels, `__init__.py`, or `layers.py`; only prefill `seq_len` varies.
- Minimize LOC and maximize execution density without obscuring necessary memory or graph-capture state. Keep logic linear, flatten conditionals, prepare iterables outside loops, and avoid redundant branches and intermediates.
- Reuse overlapping implementations by parameterizing existing functions—even with an extra flag or branch—instead of duplicating logic.
- Inline clear local helpers shorter than roughly four LOC unless they are reused pervasively across the codebase.
- Keep signatures, calls, conditions, and literals on one line to roughly 110–120 characters; when wrapping, align continuations under the opening expression rather than placing one item per line.
- While builds, tests, or commands run, continue with independent useful work; wait only when the next action depends on the result.
