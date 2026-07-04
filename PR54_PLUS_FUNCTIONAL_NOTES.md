# PR #54+ Functional Notes

Scope: audit of merged PRs #54-#60.

## Suggested follow-ups

1. Make the README vision ingest example explicitly fail-open:
   call `desc = llm.describe_image(...)`, then append the caption only when
   `desc.ok and desc.text`. The provider API already supports this, but the
   current example can encourage indexing an empty caption when a text-only or
   failed vision provider is used.

2. Record `--only-questions` metadata in LoCoMo output config:
   include the input path and selected QA count. The mode changes the evaluated
   subset, but the JSON config currently does not identify that subset, making
   later artifact comparison harder.

3. Document benchmark `degraded` semantics for empty-ground-truth QA:
   those rows skip recall, so `degraded: []` means "stack not exercised", not
   necessarily "healthy stack".

4. Consider a small helper for image-message ingest:
   e.g. `IngestItem.from_image_message(...)` or a utility that calls
   `describe_image()` and appends the caption safely. Keeping image description
   opt-in is right, but users should not need to copy glue code.

5. Sync the README fail-open wording with `ARCHITECTURE.md`:
   top-level README wording can still be read as "everything is fail-open",
   while query expansion is explicitly an exception in the architecture docs.

## Priority

First do items 1 and 2. They are small and reduce misuse / benchmark artifact
ambiguity without changing runtime behavior.
