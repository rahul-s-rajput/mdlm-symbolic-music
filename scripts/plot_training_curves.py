"""
plot_training_curves.py — Generate clean training curve figures for the report.

Reads WandB CSV exports and produces publication-quality plots of val/nll and
val/ppl for AR vs MDLM.

Usage (from duo/ directory):
    python plot_training_curves.py

Outputs:
    outputs/figures/val_nll_curves.png
    outputs/figures/val_ppl_curves.png
    outputs/figures/val_nll_ppl_combined.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────
NLL_CSV  = 'outputs/wandb_val_nll.csv'
PPL_CSV  = 'outputs/wandb_val_ppl.csv'
OUT_DIR  = 'outputs/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
AR_COLOR   = '#DD8452'   # warm orange
MDLM_COLOR = '#4C72B0'   # blue
AR_BEST    = 1.805
MDLM_BEST  = 1.741


def load_csv(path, metric):
    """
    Load WandB CSV export. Columns of interest:
      'trainer/global_step'
      'ar-midi-v2 - val/<metric>'
      'mdlm-midi-v1 - val/<metric>'
    Returns two DataFrames: (ar_df, mdlm_df) each with [step, value].
    """
    df = pd.read_csv(path)

    ar_col   = f'ar-midi-v2 - val/{metric}'
    mdlm_col = f'mdlm-midi-v1 - val/{metric}'
    step_col = 'trainer/global_step'

    ar_df = (df[[step_col, ar_col]]
             .dropna()
             .rename(columns={step_col: 'step', ar_col: 'value'})
             .drop_duplicates('step')
             .sort_values('step')
             .reset_index(drop=True))

    mdlm_df = (df[[step_col, mdlm_col]]
               .dropna()
               .rename(columns={step_col: 'step', mdlm_col: 'value'})
               .drop_duplicates('step')
               .sort_values('step')
               .reset_index(drop=True))

    return ar_df, mdlm_df


def plot_single(ar_df, mdlm_df, metric, ylabel, title, out_path,
                ar_best=None, mdlm_best=None):
    """Plot a single metric (NLL or PPL) for both models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ar_df['step'],   ar_df['value'],
            color=AR_COLOR,   lw=2,   label='AR Baseline',  zorder=3)
    ax.plot(mdlm_df['step'], mdlm_df['value'],
            color=MDLM_COLOR, lw=2,   label='MDLM',          zorder=3)

    # Mark best points
    if ar_best is not None:
        best_ar_row = ar_df.loc[ar_df['value'].idxmin()]
        ax.scatter(best_ar_row['step'], best_ar_row['value'],
                   color=AR_COLOR, s=80, zorder=5)
        ax.annotate(f'  AR best\n  {best_ar_row["value"]:.3f}',
                    xy=(best_ar_row['step'], best_ar_row['value']),
                    fontsize=8.5, color=AR_COLOR, va='top')

    if mdlm_best is not None:
        best_mdlm_row = mdlm_df.loc[mdlm_df['value'].idxmin()]
        ax.scatter(best_mdlm_row['step'], best_mdlm_row['value'],
                   color=MDLM_COLOR, s=80, zorder=5)
        ax.annotate(f'  MDLM best\n  {best_mdlm_row["value"]:.3f}',
                    xy=(best_mdlm_row['step'], best_mdlm_row['value']),
                    fontsize=8.5, color=MDLM_COLOR, va='bottom')

    ax.set_xlabel('Global Step (optimizer steps)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_combined(ar_nll, mdlm_nll, ar_ppl, mdlm_ppl, out_path):
    """Side-by-side NLL and PPL subplots — good for a paper figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # NLL
    ax1.plot(ar_nll['step'],   ar_nll['value'],
             color=AR_COLOR,   lw=2, label='AR Baseline')
    ax1.plot(mdlm_nll['step'], mdlm_nll['value'],
             color=MDLM_COLOR, lw=2, label='MDLM')

    best_ar   = ar_nll.loc[ar_nll['value'].idxmin()]
    best_mdlm = mdlm_nll.loc[mdlm_nll['value'].idxmin()]
    ax1.scatter(best_ar['step'],   best_ar['value'],
                color=AR_COLOR,   s=70, zorder=5)
    ax1.scatter(best_mdlm['step'], best_mdlm['value'],
                color=MDLM_COLOR, s=70, zorder=5)
    ax1.annotate(f'{best_ar["value"]:.3f}',
                 xy=(best_ar['step'], best_ar['value']),
                 xytext=(best_ar['step'] + 500, best_ar['value'] + 0.05),
                 fontsize=8, color=AR_COLOR)
    ax1.annotate(f'{best_mdlm["value"]:.3f}',
                 xy=(best_mdlm['step'], best_mdlm['value']),
                 xytext=(best_mdlm['step'] - 3000, best_mdlm['value'] - 0.08),
                 fontsize=8, color=MDLM_COLOR)

    ax1.set_xlabel('Global Step', fontsize=11)
    ax1.set_ylabel('Validation NLL (nats)', fontsize=11)
    ax1.set_title('Validation NLL', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    # PPL
    ax2.plot(ar_ppl['step'],   ar_ppl['value'],
             color=AR_COLOR,   lw=2, label='AR Baseline')
    ax2.plot(mdlm_ppl['step'], mdlm_ppl['value'],
             color=MDLM_COLOR, lw=2, label='MDLM')

    best_ar_p   = ar_ppl.loc[ar_ppl['value'].idxmin()]
    best_mdlm_p = mdlm_ppl.loc[mdlm_ppl['value'].idxmin()]
    ax2.scatter(best_ar_p['step'],   best_ar_p['value'],
                color=AR_COLOR,   s=70, zorder=5)
    ax2.scatter(best_mdlm_p['step'], best_mdlm_p['value'],
                color=MDLM_COLOR, s=70, zorder=5)
    ax2.annotate(f'{best_ar_p["value"]:.2f}',
                 xy=(best_ar_p['step'], best_ar_p['value']),
                 xytext=(best_ar_p['step'] + 500, best_ar_p['value'] + 0.1),
                 fontsize=8, color=AR_COLOR)
    ax2.annotate(f'{best_mdlm_p["value"]:.2f}',
                 xy=(best_mdlm_p['step'], best_mdlm_p['value']),
                 xytext=(best_mdlm_p['step'] - 3000, best_mdlm_p['value'] - 0.3),
                 fontsize=8, color=MDLM_COLOR)

    ax2.set_xlabel('Global Step', fontsize=11)
    ax2.set_ylabel('Validation Perplexity', fontsize=11)
    ax2.set_title('Validation Perplexity', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    fig.suptitle('AR vs MDLM — Training Curves on MAESTRO v3',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    print('Loading CSVs...')
    ar_nll,  mdlm_nll  = load_csv(NLL_CSV, 'nll')
    ar_ppl,  mdlm_ppl  = load_csv(PPL_CSV, 'ppl')

    print(f'AR NLL points:   {len(ar_nll)},  MDLM NLL points: {len(mdlm_nll)}')
    print(f'AR PPL points:   {len(ar_ppl)},  MDLM PPL points: {len(mdlm_ppl)}')
    print(f'AR best NLL:     {ar_nll["value"].min():.4f}')
    print(f'MDLM best NLL:   {mdlm_nll["value"].min():.4f}')
    print(f'AR best PPL:     {ar_ppl["value"].min():.4f}')
    print(f'MDLM best PPL:   {mdlm_ppl["value"].min():.4f}')

    print('\nGenerating plots...')

    plot_single(ar_nll, mdlm_nll,
                metric='nll',
                ylabel='Validation NLL (nats)',
                title='AR vs MDLM — Validation NLL on MAESTRO v3',
                out_path=f'{OUT_DIR}/val_nll_curves.png',
                ar_best=AR_BEST, mdlm_best=MDLM_BEST)

    plot_single(ar_ppl, mdlm_ppl,
                metric='ppl',
                ylabel='Validation Perplexity',
                title='AR vs MDLM — Validation Perplexity on MAESTRO v3',
                out_path=f'{OUT_DIR}/val_ppl_curves.png')

    plot_combined(ar_nll, mdlm_nll, ar_ppl, mdlm_ppl,
                  out_path=f'{OUT_DIR}/val_nll_ppl_combined.png')

    print('\nAll done! Figures saved to outputs/figures/')


if __name__ == '__main__':
    main()
