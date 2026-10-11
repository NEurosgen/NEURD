"""Scientific timing reports must preserve nesting, failed and unfinished work."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('aggregate_cgal_phases', Path(__file__).resolve().parents[1] / 'tools/aggregate_cgal_phases.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows(*values):
    return [json.dumps(dict(event='neurd_cgal_phase', **row)) for row in values]


def test_nested_durations_are_counted_once_with_roles():
    result = MODULE.summarize(rows(
        dict(phase='final', state='begin', depth=0, parent_phase=None, call_role='initial'),
        dict(phase='volume', state='begin', depth=1, parent_phase='final', call_role='initial'),
        dict(phase='volume', state='end', depth=1, seconds=7.0, status='complete', call_role='initial'),
        dict(phase='final', state='end', depth=0, seconds=10.0, status='complete', call_role='initial')))
    assert result['timed_self_seconds'] == 10.0
    assert result['phases'][0]['self_seconds'] == 7.0
    assert result['phases'][1]['self_seconds'] == 3.0
    assert result['phases'][0]['call_role'] == 'initial'


def test_error_and_interruption_are_preserved():
    result = MODULE.summarize(['not JSON', *rows(
        dict(phase='outer', state='begin'),
        dict(phase='inner', state='begin'),
        dict(phase='inner', state='end', seconds=2, status='error'))])
    assert result['status'] == 'interrupted'
    assert result['error_phase_count'] == 1
    assert result['unfinished'][0]['phase'] == 'outer'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, True, '1', None])
def test_invalid_durations_cannot_become_speed_evidence(bad):
    with pytest.raises(ValueError, match='duration'):
        MODULE.summarize(rows(dict(phase='a', state='begin'), dict(phase='a', state='end', seconds=bad)))


def test_mixed_workers_or_unmatched_ends_are_rejected():
    with pytest.raises(ValueError, match='depth'):
        MODULE.summarize(rows(dict(phase='a', state='begin', depth=1)))
    with pytest.raises(ValueError, match='Unmatched'):
        MODULE.summarize(rows(dict(phase='a', state='end', seconds=1)))


def test_impossible_nested_totals_are_rejected():
    with pytest.raises(ValueError, match='exceed'):
        MODULE.summarize(rows(dict(phase='a', state='begin'), dict(phase='b', state='begin'),
                              dict(phase='b', state='end', seconds=2, status='complete'),
                              dict(phase='a', state='end', seconds=1, status='complete')))
