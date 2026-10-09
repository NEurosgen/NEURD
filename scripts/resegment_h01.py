"""Launch the standalone batch planner without importing the heavy framework."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'neurd/h01_batch.py'),run_name='__main__')
