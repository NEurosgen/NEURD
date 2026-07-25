# Dead-code reachability — method + status

> **Status:** the §2 mark-and-sweep was executed to a **fixpoint** (commit `456cea1`): **45 module-level
> functions removed, −1569 LOC** — the mutually-referencing dead clusters that zero-ref misses
> (fork-divergence, ref-vector/skeleton up-down stats, parent/sibling-angle helpers,
> branch-neighbor/boutons/mesh-connectivity analysis). Wave-2 re-run = 0 candidates.
>
> **⚠️ "Exhausted" did not hold, in a way worth learning from.** A later sweep found **~3.5k more LOC**
> (~19% of the package): two whole unreachable modules (`concept_network_utils.py`;
> `parameter_utils.py` + `parameter_configs/`), 7 zero-ref functions that had drifted back, and 6 blocks
> of dead code parked in string literals. **Why the earlier sweep missed them:** it seeded `tests/**` as
> roots, so anything kept alive *only by a test* was invisible. `parameter_utils` was reachable solely
> from a 36-test file that guarded a module production never imports. **Correction to the method: a test
> is not a root for a production-reachability question.** Sweep with production roots only, then check
> separately whether each test still has a subject.
>
> Companions: `../../NEURD_STRUCTURE.md` (module map), `REFACTOR_PLAN.md`.

This is a **method reference** for future sweeps, not an open task. Note the scripts were never
committed — the sweep must be re-implemented from this spec, or the analyzer committed to `scripts/`
next time it is written.

## Two methods
- **Zero-ref (safe, fast, no false positives):** delete a module-level function iff its **name appears
  nowhere** — not as a reference, not as a string literal — across `neurd/**` + `tests/**` +
  `process_all_neurons.py`, except names matching the dynamic patterns (§hazards). Iterate to fixpoint.
  Safe because NEURD's dynamic dispatch (query system, `eval`, `getattr`) always goes **through the
  name as a string fragment** → whole-name-absent ⟹ uncallable ⟹ dead. **Blind spot:** cannot catch
  mutually-referencing dead clusters (each has ≥1 ref).
- **Mark-and-sweep from roots (catches clusters, riskier):** build a reference graph (nodes = functions +
  module globals; edges = every identifier **and string literal** matching a node name), seed the roots,
  propagate reachability, sweep the unreached. **Over-approximate edges when unsure** (keep a live
  function rather than delete a used one). Delete in small **bottom-up** batches, gate each.

## Roots (seed as reachable)
- `process_all_neurons.py` → `neuron.Neuron(mesh=...)` (production path; does NOT call
  `calculate_decomposition_products`).
- `neurd/segmentation_pipeline.py` → `sm.soma_indentification` + `neuron.Neuron`.
- `neuron.Neuron.__init__` (transitively: `preprocess_neuron`, `soma_extraction_utils`, `spine_utils`
  raw spines, width, concept network) + `Neuron/Limb/Branch/Soma` constructors & `@property` accessors.
- Parameter setup (UPSTREAM in `datasci_tools`): the `global_parameters_dict_<datatype>` /
  `attributes_dict_<datatype>` globals + `output_global_parameters_*`.
- `tests/integration/test_segmentation_pipeline.py` + `tests/unit/**`.
- Deliberately **off-path**: the whole `calculate_decomposition_products` stats chain (its only caller was
  removed) — a legitimate deletion target if stats aren't needed.

## Dynamic-dispatch hazards → model as edges/roots (naive static reachability is unsafe without these)
| Mechanism | Where | Graph implication |
|---|---|---|
| Query system, string names (`functions_list`) | `neuron_searching.query_neuron` / `apply_function_to_neuron` | string literal == a function name = an edge |
| pandas `df.eval()` restriction strings (reference query-output column names, `_limb_ns` stripped) | `neuron_searching` | names inside query/restriction strings = edges |
| `_limb_ns` suffix (`eval(f"lu.{k}_limb_ns")`) | `neuron_statistics`, `neuron_searching` | **protect every `*_limb_ns` as a root** |
| `getattr(module, f"...{var}...")` | was `concept_network_utils` (`all_{direction}_branches_from_branches`) — that module has since been deleted as dead; keep the rule for any new occurrence | protect `all_*_branches_from_branches` |
| Param convention (discovered by name upstream, invisible to a repo scan) | soma/spine/preprocess | **protect `^output_global_parameters` + all `*_dict_default/_h01/_microns` globals** |
| `@run_options` decorator | `neuron_searching` | does NOT register/scan → reachable only via explicit ref/string; name-absent = dead |

**Protected-name regex (extend, don't shrink):**
`(_limb_ns$ | ^all_.*_branches_from_branches$ | ^compute_ | ^synapses_ | ^current_ | ^endpoint_ | ^arg | ^output_global_parameters)`

## Operational gotchas (learned the hard way)
- **AST `FunctionDef.lineno` excludes decorators** → deleting `[lineno, end_lineno]` orphans
  `@run_options` → SyntaxError. Start at `min(d.lineno for d in node.decorator_list)`.
- **Methods live in `ClassDef.body`, not `tree.body`** — the zero-ref script only did module-level funcs.
  Methods reached via instances (`obj.method()`) look like bare names → high FP risk; rely on the
  characterization test.
- **Keep the deleter's module list in sync with the analyzer's** (a real bug: mismatched lists → reported
  same LOC forever, deleted nothing). Confirm LOC actually drop.
- Delete a closed candidate set **in one batch** (dead funcs' default args reference each other; partial
  deletion → import `NameError`). Also seed **module-level statement** names (aliases like
  `query_spine_objs = filter_spine_objs_from_restrictions`, default-arg refs) as roots — import-time-live.
- Decomposition writes ~tens of MB to `./<segment_id>/` in CWD → `chdir` to a tmp dir.

## Backstop
`tests/integration/test_segmentation_pipeline.py` runs the real decomposition (~11 min) on ONE microns
mesh. Does **not** cover: spine branches, query system, multi-soma, `calculate_decomposition_products`,
h01 params, alternate meshes → a deletion breaking only an uncovered path passes. Always run the fast gate
first (`py_compile` + `import neurd` + `pytest tests/unit/`, ~5 s).
