"""
User input and upload validation module.
"""
from __future__ import annotations
from pathlib import Path

def publish_inputs(data_dir: Path, uploads: dict[str, bytes], unit_mode: str = "decimal", acknowledge_extremes: bool = False):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, content in uploads.items():
        (data_dir / name).write_bytes(content)