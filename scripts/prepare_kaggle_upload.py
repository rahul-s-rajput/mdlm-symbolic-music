"""
Prepare a clean copy of DUO code for Kaggle dataset upload.

Run this from the duo/ directory:
    python prepare_kaggle_upload.py

Creates: kaggle_upload/duo-midi-code/ (relative to parent of duo/)
Upload that folder as a Kaggle dataset named "duo-midi-code".
"""
import shutil
import os
from pathlib import Path

SRC = Path(".")
DST = Path(r"E:\NEU\EAI6020\final\kaggle_upload\duo-midi-code")

# Files and dirs to include
INCLUDE_FILES = [
    "algo.py",
    "dataloader.py",
    "main.py",
    "metrics.py",
    "trainer_base.py",
    "utils.py",
    "requirements.txt",
]

INCLUDE_DIRS = [
    "configs",
    "integral",
    "models",
    "scripts",
]

# Clean destination
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

# Copy files
for f in INCLUDE_FILES:
    src_path = SRC / f
    if src_path.exists():
        shutil.copy2(src_path, DST / f)
        print(f"  Copied {f}")

# Copy directories
for d in INCLUDE_DIRS:
    src_path = SRC / d
    if src_path.exists():
        shutil.copytree(src_path, DST / d)
        print(f"  Copied {d}/")

print(f"\nDone! Upload folder: {DST}")
print(f"Upload this as a Kaggle dataset named 'duo-midi-code'")

# Show size
total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
print(f"Total size: {total / 1024 / 1024:.1f} MB")
