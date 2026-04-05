# CLAUDE.md

## After Any Code Changes

Run the following exactly after every code change:

```
uv run ruff check --fix . && uv run ruff format . && uv run pytest
```

Fix any errors or issues found, then rerun until everything passes cleanly.

## Engineering Standards

Act as a principal engineer. Be concise in explanations and outputs.

Optimize for correctness, clarity, maintainability, and security. Match the repo's language, architecture, conventions, and idioms. Prefer explicit, typed, modular designs over dynamic or implicit behavior. Maintain strong separation of concerns across modules. Create new modules, files, or classes when appropriate instead of adding unrelated functionality to existing files.

Prefer established, industry-grade solutions or simplified adaptations over custom implementations when they solve the problem well. Introduce helper abstractions such as functions, classes, inheritance, composition, properties, factories, generators, enums, type aliases, and other well-defined types when they improve readability, reduce duplication, or reduce verbosity.

Make incremental, low-risk changes unless a redesign is explicitly requested. Be meticulous in implementation. When implementing a plan, after each step explicitly verify that the step is fully completed and that no details were skipped.

## Typing

Prefer static typing over dynamic typing. Use explicit type hints and annotations throughout. Avoid `isinstance`, `getattr`, `setattr`, and `hasattr` when the type or structure is known and attributes can be accessed directly. Favor well-defined types over ad hoc lists and dictionaries. Avoid magic numbers and magic strings. Prefer enums and named constants over scattered constants. Use string enums, int enums, flag enums, and strategy-pattern enums where appropriate. Do not call `.value` or `str()` on a `StrEnum`, or `int()` on an `IntEnum`, when implicit conversion already provides the needed behavior.

## Imports and Resources

Keep imports minimal and only import what is necessary. Place imports at the top of the file unless there is a strong reason not to. Use generators when possible to reduce memory usage instead of materializing everything in memory at once. Manage resources responsibly and use appropriate language constructs to avoid leaks, excess memory usage, and unnecessary overhead. Apply safe, language-appropriate concurrency or parallelism only when justified by performance needs.

## Error Handling and Security

Handle errors explicitly. Never suppress, swallow, or ignore failures. Never introduce secrets, unsafe constructs, or security risks.

## Testing

Maintain comprehensive tests for every behavioral change. Update existing tests whenever behavior changes. Update prompt regression tests whenever modifying LLM prompts. Make prompt regression tests rigorous, tricky, and realistic enough to meaningfully stress prompt behavior. Keep prompts and prompt changes concise. Reduce verbosity, redundancy, and contradictions in instructions wherever possible.

## Documentation and Clarity

Document public APIs and any non-obvious logic. Use clear, self-describing identifiers. Seek clarification when requirements are incomplete or ambiguous.
