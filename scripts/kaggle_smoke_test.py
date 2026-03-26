# %% [markdown]
# # MDLM/DUO for Symbolic Music Generation — Smoke Test
# 
# EAI 6020 Final Project | Rahul Singh Rajput | Northeastern University
#
# This notebook runs a 100-step smoke test of MDLM on MAESTRO MIDI tokens
# using the DUO codebase (ICML 2025).
#
# **Kaggle setup:**
# - GPU: T4 x2 (Settings → Accelerator → GPU T4 x2)
# - Input datasets:
#   1. `duo-midi-code` — modified DUO source code
#   2. `maestro-tokenized` — pre-tokenized MAESTRO v3 chunks

# %% [markdown]
# ## 1. Setup

# %%
# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# %%
# Install missing deps (MidiTok not needed for training, just for preprocessing)
# wandb is optional for smoke test
!pip install -q omegaconf hydra-core lightning einops timm rich fsspec torchmetrics transformers datasets

# %%
# Copy DUO code to working directory and set up paths
import shutil
import os
import sys

# ── STEP 1: Discover dataset paths ──
# Kaggle mounts datasets at /kaggle/input/<dataset-slug>/
print("Available input datasets:")
for item in sorted(os.listdir("/kaggle/input")):
    full = os.path.join("/kaggle/input", item)
    print(f"  /kaggle/input/{item}/")
    if os.path.isdir(full):
        for sub in os.listdir(full)[:8]:
            print(f"    {sub}")

# ── STEP 2: Set paths (update slugs if yours are different) ──
CODE_DATASET = "/kaggle/input/duo-midi-code"
DATA_DATASET = "/kaggle/input/maestro-tokenized"
WORK_DIR = "/kaggle/working/duo"

# ── STEP 3: Copy code to writable dir ──
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

# Find the actual directory containing main.py (handles Kaggle nesting)
def find_dir_with_file(base, filename):
    """Walk through base to find the dir that directly contains filename."""
    if filename in os.listdir(base):
        return base
    for entry in os.listdir(base):
        candidate = os.path.join(base, entry)
        if os.path.isdir(candidate) and filename in os.listdir(candidate):
            return candidate
    # Deeper search
    for root, dirs, files in os.walk(base):
        if filename in files:
            return root
    return None

code_dir = find_dir_with_file(CODE_DATASET, "main.py")
if code_dir is None:
    raise FileNotFoundError(f"Cannot find main.py anywhere under {CODE_DATASET}")
print(f"\nFound code at: {code_dir}")
shutil.copytree(code_dir, WORK_DIR)

# ── STEP 4: Link data ──
data_link = os.path.join(WORK_DIR, "data", "maestro_tokenized")
os.makedirs(os.path.join(WORK_DIR, "data"), exist_ok=True)

data_dir = find_dir_with_file(DATA_DATASET, "vocab_info.json")
if data_dir is None:
    raise FileNotFoundError(f"Cannot find vocab_info.json anywhere under {DATA_DATASET}")
print(f"Found data at: {data_dir}")
os.symlink(data_dir, data_link)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

print(f"\nWorking directory: {os.getcwd()}")
print(f"main.py exists: {os.path.exists('main.py')}")
print(f"configs/ exists: {os.path.exists('configs')}")
print(f"vocab_info.json: {os.path.exists('data/maestro_tokenized/vocab_info.json')}")
print(f"Train data: {os.path.exists('data/maestro_tokenized/train')}")
print(f"Validation data: {os.path.exists('data/maestro_tokenized/validation')}")

import json
with open('data/maestro_tokenized/vocab_info.json') as f:
    print(f"Vocab info: {json.load(f)}")

# %%
# Quick sanity check on the data
import datasets as hf_datasets

ds = hf_datasets.load_from_disk("data/maestro_tokenized/train").with_format("torch")
print(f"Train chunks: {len(ds)}")
print(f"Sample shape: {ds[0]['input_ids'].shape}")
print(f"Max token ID: {ds[:100]['input_ids'].max().item()}")

# %% [markdown]
# ## 2. Smoke Test — MDLM (100 steps)

# %%
# Create the watch_folder that DUO expects for logging
os.makedirs("watch_folder", exist_ok=True)

# Run MDLM smoke test
# Key flags:
#   - data=maestro: uses our MIDI data config
#   - model=midi-small: 50M param model (12 layers, 512 hidden, 8 heads)  
#   - algo=mdlm: masked diffusion
#   - trainer.devices=1: single GPU (simpler for smoke test)
#   - trainer.precision=16: fp16 for T4 (no bf16 support)
#   - +wandb.offline=true: don't need wandb for smoke test

!python main.py \
    mode=train \
    data=maestro \
    model=midi-small \
    algo=mdlm \
    noise=log-linear \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    loader.num_workers=2 \
    trainer.max_steps=100 \
    trainer.devices=1 \
    trainer.precision=16 \
    trainer.val_check_interval=50 \
    trainer.log_every_n_steps=10 \
    trainer.num_sanity_val_steps=1 \
    sampling.steps=100 \
    +wandb.offline=true \
    wandb.name=mdlm-midi-smoke

# %% [markdown]
# ## 3. Smoke Test — AR Baseline (100 steps)

# %%
!python main.py \
    mode=train \
    data=maestro \
    model=midi-small \
    algo=ar \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    loader.num_workers=2 \
    trainer.max_steps=100 \
    trainer.devices=1 \
    trainer.precision=16 \
    trainer.val_check_interval=50 \
    trainer.log_every_n_steps=10 \
    trainer.num_sanity_val_steps=1 \
    +wandb.offline=true \
    wandb.name=ar-midi-smoke

# %% [markdown]
# ## 4. Check outputs
# If both runs complete without errors, the integration is working.
# Look for decreasing loss in the logs above.

# %%
# List any checkpoints or outputs created
for root, dirs, files in os.walk("outputs"):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {path} ({size / 1024:.0f} KB)")
