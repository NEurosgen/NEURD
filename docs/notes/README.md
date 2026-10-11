# Working notes

Finished investigations and open plans. These are working documents: they record how something was
found out and what is left, not polished reference material. The reference docs live at the repo root
([../../README.md](../../README.md), NEURD_STRUCTURE, PIPELINE, PERF, MESH_OPS).

| note | kind | state |
|---|---|---|
| [PUBLIC_RELEASE_PLAN.md](PUBLIC_RELEASE_PLAN.md) | actionable roadmap | branch inventory, experimental integration, correctness gates, optimization and public release |
| [DEAD_CODE_REACHABILITY.md](DEAD_CODE_REACHABILITY.md) | method reference | how to sweep for dead code, the dynamic-dispatch hazards, and why "exhausted" did not hold |
| [LOGIC_INVESTIGATION.md](LOGIC_INVESTIGATION.md) | findings record | a read-through of the decomposition logic (Areas A–E); the fragile `mesh_tools` surface |
| [NEURON_CLASSES_ARCHITECTURE.md](NEURON_CLASSES_ARCHITECTURE.md) | investigation + options | what the `Neuron`/`Limb`/`Branch` classes carry, and ranked ways to split them. Its tables are a pre-cleanup snapshot |
| [REFACTOR_PLAN.md](REFACTOR_PLAN.md) | open plan | current cleanup, compatibility boundaries and remaining readability work |
| [SPINE_UTILS_PLAN.md](SPINE_UTILS_PLAN.md) | open plan | current spine stages, supported raw API, validation and remaining dependency cleanup |
| [SPINE_VOLUME_QC.md](SPINE_VOLUME_QC.md) | confirmed defect and fix | strict centered closure, explicit hull fallback, versioned descriptor migration |
