# Agent-visible data

The expression payloads and pretrained reference artifact are prepared as task-scoped LFS objects for a verified private release repository. Materialize them before Docker build and verify their sizes and SHA256 values with the author-side release tooling. `classes.txt` contains only the public class vocabulary and is ordinary Git content. The deterministic source reconstruction procedure is documented in the author-side release ledger.
