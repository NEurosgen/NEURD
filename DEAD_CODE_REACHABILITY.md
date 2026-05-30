# Dead-code: deep reachability analysis — guide for future work

Status after Фаза 9 (+ manual cleanup through 2026-05-30, `neurd/` now ~24.8k LOC):
**zero-reference dead code is exhausted** in all 16 core modules
(`pytest tests/unit/` 59 passed; characterization test 4 passed). This doc explains what
was done, why more dead code probably still hides, and exactly how to find it **safely**.

See also: [DEPS_PLAN.md](DEPS_PLAN.md) §Фаза 9, [NEURD_STRUCTURE.md](NEURD_STRUCTURE.md),
[REFACTOR_PLAN.md](REFACTOR_PLAN.md) (further-cleanup plan: import graph, cycles, priorities),
[tests/integration/test_segmentation_pipeline.py](tests/integration/test_segmentation_pipeline.py)
(the safety backstop).

---

## 1. What "zero-ref" did, and its blind spot

The Фаза 9 method (safe, fast, no false positives): delete a module-level function iff
its **name appears nowhere** — not as a reference, not as a string literal — across
`neurd/**` + `tests/**` + `process_all_neurons.py`, **except** names matching dynamic
construction patterns (§4). Iterated to a fixpoint (deleting a function exposes others
whose only caller it was).

Why it's safe: NEURD dispatches functions dynamically (query system, `eval`, `getattr`),
**but always through the function's name as a string fragment**. If the *whole* name
never appears anywhere — including string literals — no dynamic path can construct a call
to it. So zero-ref ⟹ uncallable ⟹ dead.

**Blind spot:** zero-ref does NOT catch **mutually-referencing dead clusters**. If dead
function `A` calls dead `B` and `B` calls `A`, each has ≥1 reference, so neither is zero-ref,
yet the whole cluster is unreachable from any live entry point. Finding these requires
**mark-and-sweep reachability from the real roots** — the rest of this doc.

---

## 2. The method: mark-and-sweep from entry roots

1. **Build a reference graph.** Nodes = every function (module-level *and* class methods/
   properties — see §5 caveat) + every module global. Edges = for each function body, every
   name (identifier or string literal) that matches a known node name. Treat string literals
   as edges (the query system resolves names from strings — §3).
2. **Seed the roots** (§3) as reachable.
3. **Propagate** reachability over edges (BFS/DFS).
4. **Sweep:** any node not reached is a *candidate*. This finds the dead clusters zero-ref
   misses.
5. **Over-approximate edges for safety** (§4): if unsure whether a dynamic pattern can reach
   a function, treat it as reachable. Better to keep a live function than delete a used one.
6. **Verify every batch** with the fast gate + the characterization test (§6). Delete in
   small batches so a test failure is bisectable (the test is ~11 min).

The gap vs zero-ref: step 4's over-approximation means some "candidates" are actually live
via paths you couldn't prove dead. Those need either manual reading or the characterization
test to clear — which is why this phase is *harder and riskier* than zero-ref.

---

## 3. Roots (entry points) — seed these as reachable

The slim runtime has exactly one real entry, plus the param setup and the test:

- **`process_all_neurons.py`** → `neuron.Neuron(mesh=...)` in `_worker_process`. This is the
  user's production path. It does NOT call `calculate_decomposition_products`.
- **`neurd/segmentation_pipeline.py`** → `sm.soma_indentification` + `neuron.Neuron(mesh=...)`.
- **`neuron.Neuron.__init__`** (neuron.py) — the decomposition; transitively pulls in
  `preprocess_neuron`, `soma_extraction_utils`, `spine_utils` (raw spines), width, concept network.
- **Parameter setup:** `datasci_tools.module_utils.set_global_parameters_and_attributes_by_data_type`
  reads each module's `global_parameters_dict_<datatype>` / `attributes_dict_<datatype>` globals
  and calls `output_global_parameters_and_attributes_from_current_data_type`. (UPSTREAM, not in
  this repo — see §4.)
- **`tests/integration/test_segmentation_pipeline.py`** + `tests/unit/**` — anything they import/call.
- **Class entry points:** `Neuron`, `Limb`, `Branch`, `Soma` constructors and the `@property`
  accessors used during decomposition (these are reached via instances, not by name).

Anything reachable from these is LIVE. Note the deliberately-off-path code: the whole
`calculate_decomposition_products` stats chain is NOT a root (we removed its only caller);
it is only "partially repaired" and a legitimate deletion target if you decide stats aren't needed.

---

## 4. Dynamic-dispatch hazards — MUST be modeled as edges/roots

These are why naive static reachability is unsafe. Each is a place a function gets called
without a plain `module.func(...)` reference:

| # | Mechanism | Where | Implication for the graph |
|---|---|---|---|
| 1 | **Query system, string names.** `functions_list` may hold strings, e.g. `["median_mesh_center","width_new","labels_restriction"]`, resolved to `ns.<name>` callables. | `neuron_searching.query_neuron` / `generate_neuron_dataframe` / `apply_function_to_neuron` / `get_run_type` | A string literal equal to a function name = an edge. (Zero-ref already honored this.) |
| 2 | **pandas `df.eval()` restriction strings**, e.g. `"axon_label == True"`, reference DataFrame **columns**, which are query-function output names with the `_limb_ns` suffix stripped. | `neuron_searching` (`df.eval`, `map_new_limb_node_value`) | Treat names inside query/restriction strings as edges. |
| 3 | **`_limb_ns` suffix convention.** `eval(f"lu.{k}_{limb_function_append_name}")` builds `lu.<base>_limb_ns`. | `neuron_statistics`, `neuron_searching` (`limb_function_append_name = "limb_ns"`) | **Protect every `*_limb_ns` function as a root.** |
| 4 | **`getattr(module, f"...{var}...")`** name construction. | `concept_network_utils:` `getattr(cnu, f"all_{direction}_branches_from_branches")`; others on `np`/`tu`/objects | Protect `all_*_branches_from_branches`; for `getattr(np/tu, …)` the target is upstream, not ours. |
| 5 | **Parameter-system convention.** `output_global_parameters_{glia,nuclei,soma}` and the `global_parameters_dict_<datatype>` / `attributes_dict_<datatype>` module globals are discovered **by name in upstream `datasci_tools`** (invisible to a repo-only scan). | `soma_extraction_utils`, `spine_utils`, `preprocess_neuron`, … | **Protect `^output_global_parameters` and all `*_dict_default/_h01/_microns` globals as roots.** |
| 6 | **`@run_options` query decorator.** Sets `f.run_type`; it does **NOT** register the function in any registry and there is **no module scan** (`getmembers`/`globals()` iteration). | `neuron_searching` | Decorated functions are reachable ONLY via explicit reference/string (hazards 1–2), not by discovery. So a `@run_options` function with a name appearing nowhere IS dead. |

**Protected-name regex used in Фаза 9** (extend, don't shrink):
`(_limb_ns$ | ^all_.*_branches_from_branches$ | ^compute_ | ^synapses_ | ^current_ | ^endpoint_ | ^arg | ^output_global_parameters)`

---

## 5. Operational gotchas (learned the hard way)

- **AST `FunctionDef.lineno` excludes decorators.** Deleting `[lineno, end_lineno]` orphans
  `@run_options(...)` → SyntaxError. Use `min(d.lineno for d in node.decorator_list)` as start.
- **Methods live in `ClassDef.body`, not `tree.body`.** The Фаза 9 script only handled
  module-level functions. To extend to dead *methods*/*properties* (e.g. in `neuron.py`'s
  classes), walk `ClassDef.body` — and be extra careful: methods are reached via instances
  (`obj.method()`), which a name scan sees as the bare method name. High false-positive risk;
  rely on the characterization test.
- **Keep the deleter's module list in sync with the analyzer's.** A real bug this session:
  analyzer scanned files X, deleter still scanned files Y → loop reported the same LOC forever
  and deleted nothing. Always confirm the files' LOC actually drop.
- **Decomposition writes ~tens of MB scratch to `./<segment_id>/` in CWD.** The characterization
  test `chdir`s into a pytest tmp dir; ad-hoc runs should too, or clean `./12345/` after.
- Delete **bottom-up** within a file (descending line ranges) so earlier deletions don't shift
  later line numbers.

---

## 6. Safety backstop and its coverage limits

`tests/integration/test_segmentation_pipeline.py` runs the real decomposition (~11 min,
`xvfb-run meshlabserver`) and asserts the contract: somas + limbs/branches (`.mesh`+`.skeleton`).

**What it covers:** the `Neuron(mesh=)` decomposition path on ONE microns mesh.
**What it does NOT cover:** parameter-gated spine branches, the query system, multi-soma
split, `calculate_decomposition_products`, h01-specific params, alternate meshes. So a deletion
that only breaks an uncovered path will pass the test. For functions in those areas, read the
code and/or add a second fixture/path before trusting a green test.

Always also run the fast gate first (`python -m py_compile neurd/*.py`, `import neurd`,
`pytest tests/unit/`, ~5 s) — it catches syntax/import breakage instantly, before the 11-min run.

---

## 7. Suggested next steps

1. **Build the reference graph + mark-and-sweep** (§2) over the current 16 modules, with §3 roots
   and §4 over-approximation. Output: unreachable clusters.
2. Triage clusters: those whose names also never appear in any string and aren't param/query
   conventions are high-confidence — delete in small batches, fast-gate + characterization each.
3. Consider whether `calculate_decomposition_products` (and its stats chain) should be deleted
   outright rather than left "partially repaired" — it's off the slim path.
4. Only then consider **untangling the 18 module-load cycles** (a refactor, no LOC win) — lift
   shared helpers/types into a base module so the bottom-of-file imports around `neuron_utils`
   can move to the top.
