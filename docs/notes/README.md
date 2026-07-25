# Working notes

Finished investigations and open plans. These are working documents: they record how something was
found out and what is left, not polished reference material. The reference docs live at the repo root
([../../README.md](../../README.md), NEURD_STRUCTURE, PIPELINE, PERF, MESH_OPS).

| note | kind | state |
|---|---|---|
| [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md) | method reference | how to sweep for dead code, the dynamic-dispatch hazards, and why "exhausted" did not hold |
| [LOGIC_INVESTIGATION.md](LOGIC_INVESTIGATION.md) | findings record | a read-through of the decomposition logic (Areas A–E); the fragile `mesh_tools` surface |
| [NEURON_CLASSES_ARCHITECTURE.md](NEURON_CLASSES_ARCHITECTURE.md) | investigation + options | what the `Neuron`/`Limb`/`Branch` classes carry, and ranked ways to split them. Its tables are a pre-cleanup snapshot |
| [REFACTOR_PLAN.md](REFACTOR_PLAN.md) | open plan | the readability pass: what is done, what remains, and the AST method for `verbose` cleanup |
| [SPINE_UTILS_PLAN.md](SPINE_UTILS_PLAN.md) | open plan | migrating `spine_utils` off `mesh_tools` — read the corrections at the top before starting |
