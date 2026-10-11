"""Parallel raw-neuron runner using the same workers and resume checks as the API.

The concurrent input-size budget counts file bytes. It is an admission heuristic,
not a RAM limit; SEG_RSS_LIMIT_MB enforces observed RSS separately.
"""
import argparse
import os

from process_all_neurons import add_batch_arguments, process_folder_with_manifest
from neurd.run_config import config_from_args

DEFAULT_INPUT_SIZE_BUDGET_MB = 1024
DEFAULT_MEM_BUDGET_MB = DEFAULT_INPUT_SIZE_BUDGET_MB  # Compatibility alias.


def process_folder_parallel(
    input_dir, output_dir, manifest="processed.txt", pattern="*.off",
    no_save=False, no_decimate=False, max_workers=None,
    mem_budget_mb=DEFAULT_INPUT_SIZE_BUDGET_MB, poll_interval=0.2, *,
    input_size_budget_mb=None, config=None, retry_failed=False,
    decimation_ratio=0.0, max_somas=None,
):
    """Schedule isolated workers; ``mem_budget_mb`` is a legacy file-size alias."""
    budget = mem_budget_mb if input_size_budget_mb is None else input_size_budget_mb
    return process_folder_with_manifest(
        input_dir, output_dir, manifest=manifest, pattern=pattern,
        no_save=no_save, no_decimate=no_decimate, max_workers=max_workers,
        input_size_budget_mb=budget, poll_interval=poll_interval, config=config,
        retry_failed=retry_failed, decimation_ratio=decimation_ratio, max_somas=max_somas,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_batch_arguments(parser)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("SEG_MAX_WORKERS", "0")),
                        help="maximum concurrent workers (0 = number of cores)")
    parser.add_argument(
        "--input-size-budget-mb", "--mem-budget-mb", dest="input_size_budget_mb", type=int,
        default=int(os.environ.get("SEG_INPUT_SIZE_BUDGET_MB", os.environ.get("SEG_MEM_BUDGET_MB", "1024"))),
        help="combined concurrent input-file size in MiB; this is not a RAM limit",
    )
    args = parser.parse_args(argv)
    try:
        result = process_folder_parallel(
            args.input_dir, args.output_dir, manifest=args.manifest, pattern=args.pattern,
            no_save=args.no_save, no_decimate=args.no_decimate, config=config_from_args(args),
            retry_failed=args.retry_failed, decimation_ratio=args.decimation_ratio,
            max_somas=args.max_somas, max_workers=args.max_workers,
            input_size_budget_mb=args.input_size_budget_mb,
        )
        return result.exit_code
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[x] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
