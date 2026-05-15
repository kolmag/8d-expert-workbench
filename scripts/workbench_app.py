"""Compatibility launcher for the root workbench app."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "workbench_app.py"), run_name="__main__")
