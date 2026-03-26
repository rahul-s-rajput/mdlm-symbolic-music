# MDLM for Symbolic Music Generation

> **Forked from [s-sahoo/duo](https://github.com/s-sahoo/duo).**
> This fork applies Masked Diffusion Language Models (MDLM) and an Autoregressive (AR) Transformer baseline to symbolic music generation on MAESTRO v3, as part of EAI 6020: AI System Technologies at Northeastern University (Spring 2026).

---

## Overview

This project is the first application of MDLM (Sahoo et al., NeurIPS 2024) to symbolic music generation using MIDI tokens. MDLM's bidirectional masked diffusion is compared against an AR Transformer baseline under identical model capacity and training conditions.

**Key results:**

| Model | Params | Best Val NLL | Best Val PPL |
|-------|--------|-------------|-------------|
| AR Baseline | 38.1M | 1.805 | 6.08 |
| MDLM | 43.1M | **1.741** | **5.70** |

MDLM also uniquely enables **musical infilling** (reconstructing masked middle sections of real pieces), achieving PCHO 0.855 and groove similarity 0.640 over 50 MAESTRO validation pieces.

---

## Repository Structure

```
duo/
├── configs/                  # Hydra configs (data, model, algo, etc.)
│   ├── config.yaml
│   ├── data/maestro.yaml     # ← added: MAESTRO dataset config
│   ├── model/midi-small.yaml # ← added: small DIT config for MIDI
│   └── ...
├── models/                   # DIT backbone, EMA, UNet
│   └── dit.py                # ← modified: conditional flash-attn import
├── scripts/                  # ← added: all project-specific scripts
│   ├── preprocess_maestro.py # tokenize + chunk MAESTRO → HuggingFace dataset
│   ├── generate_samples.py   # unconditional generation (AR + MDLM)
│   ├── infilling.py          # MDLM infilling experiment
│   ├── evaluate_metrics.py   # MusPy music quality metrics
│   ├── evaluate_infilling.py # PCHO / groove / NDR infilling metrics
│   ├── midi_to_audio.py      # MIDI → WAV via tinysoundfont
│   ├── plot_training_curves.py # training curve plots from WandB CSVs
│   └── sanity_check.py       # quick data/model sanity check
├── outputs/
│   ├── figures/              # training curve PNGs
│   ├── metrics_comparison.png
│   ├── metrics_results.json
│   ├── infilling_metrics.json
│   ├── wandb_val_nll.csv
│   └── wandb_val_ppl.csv
├── algo.py                   # ← modified: AR.nll() signature fix
├── dataloader.py             # ← modified: MidiTokenizer + MAESTRO loading
├── main.py                   # ← modified: weights_only monkey-patch
├── trainer_base.py
├── metrics.py
├── utils.py
├── DEVLOG.md                 # full change log used for the report
├── requirements.txt          # core dependencies
├── requirements-midi.txt     # MIDI-specific dependencies
└── LICENSE
```

**Files modified from upstream DUO:**
- `algo.py` — added `**kwargs` to `AR.nll()` to match `trainer_base._loss()` signature
- `dataloader.py` — added `MidiTokenizer` class and MAESTRO dataset loading
- `main.py` — monkey-patch for PyTorch 2.6 `weights_only=True` checkpoint loading
- `models/dit.py` — conditional `flash_attn` import with SDPA fallback for T4 GPUs
- `configs/data/maestro.yaml` — new config for MAESTRO dataset
- `configs/model/midi-small.yaml` — new 12L/512d/8h DIT config (~43M params)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mdlm-symbolic-music.git
cd mdlm-symbolic-music

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install MIDI dependencies
pip install -r requirements-midi.txt

# 4. Download MAESTRO v3
#    https://magenta.tensorflow.org/datasets/maestro
#    Extract to data/maestro-v3.0.0/

# 5. Tokenize and chunk
python scripts/preprocess_maestro.py \
    --maestro_dir data/maestro-v3.0.0 \
    --output_dir data/maestro_tokenized

# 6. (Optional) Download soundfont for audio rendering
#    Get GeneralUser GS from https://schristiancollins.com/generaluser.php
#    Place as GeneralUser-GS.sf2 in the project root
```

---

## Training

```bash
# AR Baseline
python main.py \
    data=maestro \
    model=midi-small \
    algo=ar \
    trainer.max_steps=25000 \
    attn_backend=sdpa

# MDLM
python main.py \
    data=maestro \
    model=midi-small \
    algo=mdlm \
    trainer.max_steps=25000 \
    sampling.predictor=ancestral_cache \
    attn_backend=sdpa
```

**Hardware used:**
- AR: Kaggle 2×T4 (~12 hrs)
- MDLM: Azure 1×T4 (~20 hrs)

---

## Evaluation

```bash
# Generate samples (adjust --checkpoint path)
python scripts/generate_samples.py \
    --model ar   --checkpoint ar_best.ckpt   --num_samples 50
python scripts/generate_samples.py \
    --model mdlm --checkpoint mdlm_best.ckpt --num_samples 50

# MusPy music quality metrics
python scripts/evaluate_metrics.py

# MDLM infilling
python scripts/infilling.py \
    --checkpoint mdlm_best.ckpt --num_pieces 50

# Infilling quality metrics (PCHO / groove / NDR)
python scripts/evaluate_infilling.py

# Convert MIDI to audio
python scripts/midi_to_audio.py --soundfont GeneralUser-GS.sf2

# Plot training curves (from WandB CSV exports)
python scripts/plot_training_curves.py
```

---

## Results

### Training Curves

![Training Curves](outputs/figures/val_nll_ppl_combined.png)

### Music Quality Metrics (n=50 per model)

![Metrics Comparison](outputs/metrics_comparison.png)

### Infilling Quality (n=50 MAESTRO validation pieces)

| Metric | Mean | Std |
|--------|------|-----|
| Pitch Class Histogram Overlap (PCHO) | 0.855 | ±0.107 |
| Groove Similarity | 0.640 | ±0.155 |
| Note Density Ratio | 0.968 | ±0.381 |

---

## Key Bugs Fixed (vs upstream DUO)

1. MidiTok v3: `special_tokens_ids` is a list, not a dict — use `tokenizer["PAD_None"]`
2. Windows: `os.sched_getaffinity` not available — added `_get_num_proc()` fallback in `dataloader.py`
3. T4 GPU: `flash_attn` import crash — conditional import with SDPA fallback in `models/dit.py`
4. AR NLL signature mismatch — added `**kwargs` to `AR.nll()` in `algo.py`
5. PyTorch 2.6 checkpoint loading — monkey-patch `weights_only=True` in `main.py`

---

## Citation

If you use this code, please cite the original DUO paper:

```bibtex
@inproceedings{sahoo2025duo,
  title={The Diffusion Duality},
  author={Sahoo, Subham and others},
  booktitle={ICML},
  year={2025}
}
```

And the MDLM paper:

```bibtex
@inproceedings{sahoo2024mdlm,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham and others},
  booktitle={NeurIPS},
  year={2024}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE). This fork inherits the upstream license.
