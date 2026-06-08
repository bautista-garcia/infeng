- **Indentation Style:** Use compact continuation indentation: keep signatures, calls, conditions, and data literals on one line up to roughly 110-120 chars; when wrapping, align continuation lines under the opening expression instead of one-item-per-line formatting.

### Minimize Lines of Code (LOC)
* **Principle:** Fewer lines of code are strictly superior. 
* **Rationale:** The ultimate bottleneck in software engineering is the amount of context a reader must maintain in working memory. Minimizing LOC directly minimizes the state space and logic space a developer must track to comprehend a file or module.
* **Execution:** Optimize for high-density, concise expressions. Avoid verbose configurations, boilerplate, or multi-line setups when a compact form is expressive and correct.

### Maximize Function Reuse via Parameterization
* **Principle:** Reuse existing functions over writing new ones, even if reuse requires introducing an extra conditional branch or parameter.
* **Rationale:** Consolidating behavior into a single focal point prevents logic duplication and forces a unified interface. A new branch inside a well-tested function is easier to audit than a separate, nearly identical function.
* **Execution:** When implementing a feature that overlaps 80% with an existing function, do not duplicate it. Add a control flag, optional parameter, or branching logic to the existing function to accommodate the new behavior.

### Eliminate Low-Utility Helper Functions (Prefer Inline Logic)
* **Principle:** Do not extract short helper functions (typically < 4 LOC) unless they are used pervasively across the entire codebase.
* **Rationale:** Offloading small chunks of code to a helper creates artificial abstraction layers and forces the reader to jump around the file (increasing context switches). If a short sequence of logic is local or used only in a few spots, inline it.
* **Execution:** Keep logic linear and local. If a block of code is short and its purpose is clear from the context, write it inline where it executes. Do not wrap it in a micro-function.

<parallel-work>
Never wait idly for a command, build, test, or background job to finish. While it runs, do useful parallel work: review code, simplify, delete unnecessary pieces, rethink the approach, inspect failures, deepen tests, or think through the next move. Only block when the next action truly depends on that result.
</parallel-work>

- **High-Density, Minimalist Code:** Prioritize low Lines of Code (LOC) and maximum tensor execution density. Flatten nested conditional logic using compact idioms (e.g., `getattr(obj, "attr", default)` inside single-line generators). Eliminate redundant runtime branching inside loops by pre-allocating or structuring iterables beforehand (e.g., using `zip()` over pre-arranged layouts instead of executing ternary checks at every iteration). Avoid loose intermediate variables unless strictly necessary for memory tracking or graph capture.
