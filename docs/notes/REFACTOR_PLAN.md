# Readability refactor — plan and method

Goal: remove clutter without changing behaviour. One file per commit.
Backstop: `pytest` (the fast gate) after each file, plus
`python tests/tools/neuron_gate.py --baseline tests/fixtures/neuron_baseline_2530864375.json --tolerance`
(~3.5 min) for anything on the decomposition path.

The earlier structural refactor (import cycles, dead code) is in `git log`. This document is about
readability only.

> **Status.** The small and medium files in §2 are done: `width_utils` (457→183), `branch_utils`
> (782→355, plus the skeletal-coords decoupling), `neuron_statistics` (→347). `limb_utils` and
> `concept_network_utils` no longer exist (the latter was deleted as dead). The large files are
> partially done: `neuron.py`'s classes are decomposed (3403→1825) and `preprocess_neuron` is phased
> (→3174). What remains: `spine_utils` (3136 — off the segmentation path, needs a spine harness, see
> [SPINE_UTILS_PLAN.md](SPINE_UTILS_PLAN.md)) and the tails of `neuron_utils` (2599) /
> `preprocess_neuron`. The §1 smell table and the §2⑤ AST method are durable — keep applying them.

---

## 1. Recurring smells (present in every file)

| smell | example | what to do |
|---|---|---|
| **Bottom imports** | `#--- from neurd_packages ---` at the end of the file | hoist to the header |
| **Duplicate imports** | `numpy_dep as np` on both line 3 and line 107 | keep one |
| **Dead imports** | `import time` / `ipyvolume_utils`, unused | delete |
| **`verbose` / `print_flag` spam** | `if verbose: print(f"...")` in 5–10 places in one function | delete the prints; drop `verbose` from the signature |
| **Commented-out code** | `# if print_flag:` / `# print(f"...")` | delete |
| **No-op assignment** | `distance_threshold = distance_threshold` | delete |
| **Multi-line "example" docstrings** | 15+ lines showing a call | replace with a one-line "what it does" |
| **Empty blocks** | 10+ consecutive blank lines | collapse to one |
| **Dead code parked in a string literal** | `''' Old method ... '''` around a full function body | delete (keep genuine design notes, but as comments) |

---

## 2. Files, in priority order

### ①②③ `limb_utils` / `width_utils` (457→183) / `branch_utils` (782→355) ✅ DONE

The small and medium files are cleaned (dead/duplicate imports, bottom imports hoisted,
`verbose`/`print_flag` spam, long example docstrings). `limb_utils` is no longer a separate file;
`branch_utils` additionally got the skeletal-coords decoupling.

### ④ `concept_network_utils.py` — 1325 → 849 LOC → **deleted**

Cleaned first (imports hoisted, all `verbose` spam removed, 4 dead `'''` blocks of commented-out
functions gone, ~227 lines), and later **deleted entirely**: the whole module turned out to be
unreachable — its only mention repo-wide was a comment. A reminder that cleaning a module is not
evidence that it is used.

### ④·5 `neuron_statistics.py` → 347 LOC ✅ DONE

Imports hoisted (dedup `gu`/`np`, `numpy_dep`→`numpy`; `nst` stays a local cycle-breaker). Removed 33
pure `if verbose: print` blocks and their pass-throughs; the `verbose` parameter was dropped from 19
functions and kept on 7 that genuinely use it. Method: an **AST transformer** (below), safer than
regexes.

Note the module is now reachable only via `Neuron.neuron_stats` → `calculate_decomposition_products`,
which the slim path never calls. It is kept because `neuron_stats` is public API, so removing it is an
API break rather than a dead-code deletion.

### ⑤ The large files (a separate session)

`neuron_utils.py` (2599), `preprocess_neuron.py` (3174 — already decomposed into named phases, see
PIPELINE.md §2), `spine_utils.py` (3136). Higher risk: the critical pipeline path.

Also open here: **the ~20 bottom-of-file imports** that remain as circular-import workarounds
(`spine_utils.py` has them in its last 25 lines; also `neuron.py`, `neuron_searching.py`,
`neuron_utils.py`, `soma_extraction_utils.py`), plus a handful of per-call local imports. Hoisting them
requires actually breaking the cycles, not just moving lines.

**AST method for verbose cleanup** (developed on `concept_network`/`neuron_statistics`):

1. `tokenize` → collect the line numbers inside multi-line strings (docstrings, `'''` blocks) and never
   touch them.
2. `ast`: delete `if verbose:` blocks **only** when the body is a "pure print" — recursively, only
   Expr-print/Constant/Pass and loops over those; never Assign/Return/Raise/With/Try.
3. Drop the `verbose` parameter from a function **only** if no `verbose` (Load) remains in its body
   after the blocks and pass-throughs are removed. Otherwise keep it.
4. Final AST check: every remaining `verbose` (Load) has a parameter in its own function — this catches
   a `NameError` before the tests do. Then run the unit gate and the decomposition gate.

Backstop combo: `py_compile` + the AST scope check + `pytest` + the decomposition gate.

---

## 3. Conventions

- **Don't change behaviour.** Renames and dead-code deletion are fine. Logic is not.
- **One file per commit:** `refactor(readability): <file> — what was removed`.
- **Fast gate after every commit** (`pytest`).
- **Remove `verbose`/`print_flag` entirely**, including the signature parameter — callers pass it
  through `**kwargs` and will not notice.
- **Bottom imports → header**, dropping the separator comment — but only where no cycle depends on the
  placement.
- **`__init__.py` needs care.** Its compat and perf patches are fragile and each block has its own
  reason. It is not off-limits (six perf patches were added there deliberately), but any change must
  keep `tests/unit/test_perf_patches.py` green — that test asserts every patch actually installed.
