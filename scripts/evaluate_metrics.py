"""
evaluate_metrics.py — Automatic music quality metrics on generated samples.

Uses muspy's verified API (v0.5.0):
  muspy.pitch_class_entropy(music)
  muspy.pitch_range(music)
  muspy.n_pitches_used(music)
  muspy.polyphony(music)
  muspy.polyphony_rate(music, threshold=2)
  muspy.empty_beat_rate(music)
  muspy.groove_consistency(music, measure_resolution)

Usage (from duo/ directory):
    python evaluate_metrics.py

Requirements:
    pip install muspy matplotlib
"""

import glob
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np


def load_midi_as_muspy(midi_path):
    """Load a MIDI file using muspy."""
    import muspy
    try:
        music = muspy.read_midi(midi_path)
        return music
    except Exception as e:
        print(f"    Warning: could not load {os.path.basename(midi_path)}: {e}")
        return None


def compute_metrics_for_file(midi_path):
    """Compute all muspy metrics for a single MIDI file using verified API."""
    import muspy
    import math

    music = load_midi_as_muspy(midi_path)
    if music is None:
        return None

    # Check there are actually notes
    total_notes = sum(len(t.notes) for t in music.tracks)
    if total_notes == 0:
        print(f"    Warning: no notes found in {os.path.basename(midi_path)}")
        return None

    metrics = {}

    def safe(fn, *args, **kwargs):
        """Call a muspy function safely, returning None on error or NaN."""
        try:
            val = fn(*args, **kwargs)
            if val is None:
                return None
            try:
                if math.isnan(float(val)):
                    return None
            except (TypeError, ValueError):
                pass
            return float(val)
        except Exception as e:
            return None

    # Pitch class entropy — Shannon entropy over 12 pitch classes (0–3.58)
    metrics['pitch_class_entropy'] = safe(muspy.pitch_class_entropy, music)

    # Pitch range — highest minus lowest pitch in semitones
    metrics['pitch_range'] = safe(muspy.pitch_range, music)

    # Number of unique pitches used (proxy for melodic diversity)
    metrics['n_pitches_used'] = safe(muspy.n_pitches_used, music)

    # Polyphony — avg concurrent pitches when at least one note is on
    metrics['polyphony'] = safe(muspy.polyphony, music)

    # Polyphony rate — fraction of time steps with >1 simultaneous note
    metrics['polyphony_rate'] = safe(muspy.polyphony_rate, music, threshold=2)

    # Empty beat rate — fraction of beats with no notes (lower = more notes)
    metrics['empty_beat_rate'] = safe(muspy.empty_beat_rate, music)

    # Groove consistency — rhythmic regularity across measures
    # measure_resolution = resolution (ticks per beat) since 4/4 = 4 beats/measure
    # muspy default resolution is 24 ticks/beat, so measure = 96 ticks
    measure_resolution = music.resolution * 4
    metrics['groove_consistency'] = safe(
        muspy.groove_consistency, music, measure_resolution)

    # Filter out None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    if metrics:
        summary = ', '.join(f'{k}={v:.3f}' for k, v in metrics.items())
        print(f"    {summary}")

    return metrics if metrics else None


def compute_metrics_for_model(midi_dir, model_name, exclude_patterns=None):
    """Compute average metrics across all MIDI files for a model."""
    midi_files = sorted(glob.glob(os.path.join(midi_dir, '*.mid')))

    # Exclude certain file patterns (e.g. masked files in infilling)
    if exclude_patterns:
        midi_files = [
            f for f in midi_files
            if not any(p in os.path.basename(f) for p in exclude_patterns)
        ]

    if not midi_files:
        print(f"  No MIDI files found in {midi_dir}")
        return None

    print(f"\n{model_name} ({len(midi_files)} files):")
    all_metrics = []
    for f in midi_files:
        print(f"  {os.path.basename(f)}:")
        m = compute_metrics_for_file(f)
        if m is not None:
            all_metrics.append(m)

    if not all_metrics:
        print(f"  No valid metrics computed for {model_name}")
        return None

    # Collect all keys across all files
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    # Average across files (only files that have each metric)
    avg = {}
    for key in sorted(all_keys):
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg[key] = round(float(np.mean(values)), 4)
            avg[key + '_std'] = round(float(np.std(values)), 4)

    return avg


def print_comparison_table(results):
    """Print a formatted comparison table."""
    metrics_display = [
        ('pitch_class_entropy', 'Pitch Class Entropy', 'max 3.58'),
        ('pitch_range',         'Pitch Range',         'semitones'),
        ('n_pitches_used',      'Unique Pitches',      'count'),
        ('polyphony',           'Polyphony',           'avg concurrent'),
        ('polyphony_rate',      'Polyphony Rate',      '0-1'),
        ('empty_beat_rate',     'Empty Beat Rate',     'lower=more notes'),
        ('groove_consistency',  'Groove Consistency',  'higher=regular'),
    ]

    models = list(results.keys())
    col_w = 20
    print("\n" + "=" * (28 + col_w * len(models)))
    header = f"{'Metric':<26} " + " ".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print("=" * (28 + col_w * len(models)))

    for key, display_name, unit in metrics_display:
        row = f"{display_name:<26} "
        for model in models:
            if results[model] and key in results[model]:
                val = results[model][key]
                std = results[model].get(key + '_std', 0)
                cell = f"{val:.3f}±{std:.3f}"
                row += f"{cell:>{col_w}} "
            else:
                row += f"{'N/A':>{col_w}} "
        print(row)

    print("=" * (28 + col_w * len(models)))


def plot_comparison(results, output_path='outputs/metrics_comparison.png'):
    """
    Generate two side-by-side subplots:
      Left  — count-scale metrics (pitch range, unique pitches)
      Right — normalised metrics (entropy, polyphony, rates, groove)
    Saves to output_path.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot. Run: pip install matplotlib")
        return

    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("No valid results to plot")
        return

    models = list(valid_results.keys())
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    # ── Two groups of metrics ─────────────────────────────────────────────
    count_metrics = [
        ('pitch_range',    'Pitch Range\n(semitones)'),
        ('n_pitches_used', 'Unique Pitches'),
    ]
    norm_metrics = [
        ('pitch_class_entropy', 'Pitch Class\nEntropy'),
        ('polyphony',           'Polyphony'),
        ('polyphony_rate',      'Polyphony\nRate'),
        ('empty_beat_rate',     'Empty Beat\nRate'),
        ('groove_consistency',  'Groove\nConsistency'),
    ]

    def draw_group(ax, metric_list, title):
        x = np.arange(len(metric_list))
        width = 0.75 / max(len(models), 1)
        for i, (model, color) in enumerate(zip(models, colors)):
            vals, errs = [], []
            for key, _ in metric_list:
                v = valid_results[model].get(key, 0) or 0
                e = valid_results[model].get(key + '_std', 0) or 0
                vals.append(v)
                errs.append(e)
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model, color=color,
                   alpha=0.85, yerr=errs, capsize=4,
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metric_list], fontsize=10)
        ax.set_title(title, fontsize=11, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig, (ax_count, ax_norm) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={'width_ratios': [2, 5]}
    )

    draw_group(ax_count, count_metrics,  'Count-Scale Metrics')
    ax_count.set_ylabel('Count', fontsize=11)

    draw_group(ax_norm,  norm_metrics,   'Normalised Metrics')
    ax_norm.set_ylabel('Score', fontsize=11)

    fig.suptitle('AR vs MDLM — Music Quality Metrics on MAESTRO', fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()


def save_results_json(results, output_path='outputs/metrics_results.json'):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved: {output_path}")


def main():
    try:
        import muspy
    except ImportError:
        print("muspy not installed. Run: pip install muspy")
        sys.exit(1)

    # Define evaluation targets
    eval_targets = {
        'AR':   ('outputs/generated/ar',   None),
        'MDLM': ('outputs/generated/mdlm', None),
    }

    # Add real MAESTRO pieces from infilling (originals only)
    if os.path.exists('outputs/infilling'):
        originals = glob.glob('outputs/infilling/*_original.mid')
        if originals:
            eval_targets['MAESTRO (Real)'] = (
                'outputs/infilling', ['masked', 'infilled'])

    print("Computing MusPy metrics...")
    results = {}
    for model_name, (midi_dir, exclude) in eval_targets.items():
        if os.path.exists(midi_dir):
            results[model_name] = compute_metrics_for_model(
                midi_dir, model_name, exclude_patterns=exclude)
        else:
            print(f"  Skipping {model_name}: {midi_dir} not found")

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("\nNo valid results computed.")
        print("Check that MIDI files exist in outputs/generated/ar/ and outputs/generated/mdlm/")
        sys.exit(1)

    print_comparison_table(valid)
    save_results_json(valid)
    plot_comparison(valid)

    print("\nDone! Check outputs/metrics_results.json and outputs/metrics_comparison.png")


if __name__ == '__main__':
    main()
