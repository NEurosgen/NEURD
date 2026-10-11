"""Summarize one worker's phase log without double-counting nested timings.

Usage: python tests/tools/aggregate_cgal_phases.py worker.log --output phases.json
Ordinary stdout/stderr is ignored. Interrupted phases and errors remain visible.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def summarize(lines):
    stack, errors, completed = [], [], []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(row, dict) or row.get('event') != 'neurd_cgal_phase':
            continue
        name = row.get('phase')
        state = row.get('state')
        if not isinstance(name, str) or state not in {'begin', 'end'}:
            raise ValueError(f'Invalid phase record at line {line_number}')
        if state == 'begin':
            if 'depth' in row and row['depth'] != len(stack):
                raise ValueError(f'Unexpected phase depth at line {line_number}; use one worker log')
            if 'parent_phase' in row and row['parent_phase'] != (stack[-1]['row']['phase'] if stack else None):
                raise ValueError(f'Unexpected parent phase at line {line_number}')
            stack.append(dict(row=row, children_seconds=0.0, line=line_number))
            continue
        if not stack or stack[-1]['row']['phase'] != name:
            raise ValueError(f'Unmatched phase end at line {line_number}')
        if 'depth' in row and row['depth'] != len(stack) - 1:
            raise ValueError(f'Unexpected end depth at line {line_number}')
        frame = stack.pop()
        seconds = row.get('seconds')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid phase duration at line {line_number}')
        own_seconds = seconds - frame['children_seconds']
        if own_seconds < -1e-6:
            raise ValueError(f'Child timings exceed parent duration at line {line_number}')
        result = dict(row, self_seconds=max(0.0, own_seconds), begin_line=frame['line'])
        completed.append(result)
        if row.get('status') != 'complete':
            errors.append(result)
        if stack:
            stack[-1]['children_seconds'] += seconds
    aggregate = defaultdict(lambda: dict(calls=0, inclusive_seconds=0.0, self_seconds=0.0, errors=0))
    for row in completed:
        key = (row['phase'], row.get('call_role'))
        value = aggregate[key]
        value['calls'] += 1
        value['inclusive_seconds'] += row['seconds']
        value['self_seconds'] += row['self_seconds']
        value['errors'] += row.get('status') != 'complete'
    groups = [dict(phase=phase, call_role=role, **value) for (phase, role), value in aggregate.items()]
    groups.sort(key=lambda row: (-row['self_seconds'], row['phase'], str(row['call_role'])))
    return dict(status='interrupted' if stack else ('completed_with_errors' if errors else 'complete'),
                measured_scope='Single worker phase records. Inclusive durations overlap; self durations exclude timed direct children. Untimed work is absent.',
                timed_self_seconds=sum(row['self_seconds'] for row in completed),
                completed_phase_count=len(completed), error_phase_count=len(errors),
                unfinished=[dict(**frame['row'], begin_line=frame['line']) for frame in stack],
                phases=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.log.open() as stream:
        result = summarize(stream)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
