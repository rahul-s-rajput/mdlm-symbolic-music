"""
evaluate_infilling.py — Quantitative evaluation of MDLM infilling quality.

Metrics (standard in symbolic music infilling literature, e.g. MIDI-RWKV 2025):
  1. Pitch Class Histogram Overlap (PCHO) — cosine similarity between
     pitch class histograms of infilled vs original region. Measures
     tonal/harmonic coherence. Range 0-1, higher = better.
  2. Groove Similarity — 1 - normalized hamming distance of onset vectors
     between infilled and original region. Measures rhythmic consistency.
     Range 0-1, higher = better.
  3. Note Density Ratio (NDR) — ratio of note density in infilled region
     to original region. Should be close to 1.0.

For each piece we compare the MIDDLE region (masked/infilled portion) of:
  - original: ground truth from MAESTRO
  - infilled: MDLM's reconstruction

Usage (from duo/ directory):
    python evaluate_infilling.py

Outputs:
    outputs/infilling_metrics.json
    outputs/figures/infilling_metrics.png
"""

import glob
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_midi_notes(midi_path):
    """Load MIDI file and return list of (pitch, start_time, end_time) tuples."""
    import muspy
    try:
        music = muspy.read_midi(midi_path)
        notes = []
        for track in music.tracks:
            for note in track.notes:
                notes.append((note.pitch, note.time, note.time + note.duration))
        return notes, music.resolution
    except Exception as e:
        print(f"  Warning: could not load {os.path.basename(midi_path)}: {e}")
        return None, None


def extract_middle_region(notes, total_duration, context_frac=0.25, suffix_frac=0.25):
    """
    Extract notes from the middle region of the piece.
    The infilling covers the middle 50% (context=25%, suffix=25%).
    We evaluate only notes that fall entirely within this middle region.
    """
    if not notes:
        return []
    start = total_duration * context_frac
    end   = total_duration * (1 - suffix_frac)
    return [(p, s, e) for (p, s, e) in notes if s >= start and e <= end]


def pitch_class_histogram(notes):
    """Compute normalized 12-bin pitch class histogram."""
    hist = np.zeros(12, dtype=np.float32)
    for (pitch, _, _) in notes:
        hist[pitch % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pcho(orig_notes, infill_notes):
    """
    Pitch Class Histogram Overlap — cosine similarity between
    pitch class histograms of original and infilled regions.
    """
    h1 = pitch_class_histogram(orig_notes)
    h2 = pitch_class_histogram(infill_notes)
    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    if norm1 == 0 or norm2 == 0:
        return None
    return float(np.dot(h1, h2) / (norm1 * norm2))


def groove_similarity(orig_notes, infill_notes, resolution, total_duration,
                      context_frac=0.25, suffix_frac=0.25):
    """
    Groove similarity — 1 - normalized hamming distance between
    binary onset vectors of original and infilled regions.
    Resolution: 16th note grid (resolution/4 ticks per 16th note).
    """
    if not orig_notes or not infill_notes:
        return None

    start = total_duration * context_frac
    end   = total_duration * (1 - suffix_frac)
    region_len = end - start

    if region_len <= 0:
        return None

    ticks_per_16th = max(1, resolution // 4)
    n_slots = max(1, int(region_len / ticks_per_16th))

    orig_vec   = np.zeros(n_slots, dtype=np.uint8)
    infill_vec = np.zeros(n_slots, dtype=np.uint8)

    for (_, onset, _) in orig_notes:
        slot = int((onset - start) / ticks_per_16th)
        if 0 <= slot < n_slots:
            orig_vec[slot] = 1

    for (_, onset, _) in infill_notes:
        slot = int((onset - start) / ticks_per_16th)
        if 0 <= slot < n_slots:
            infill_vec[slot] = 1

    hamming = float(np.sum(orig_vec != infill_vec)) / n_slots
    return 1.0 - hamming


def note_density_ratio(orig_notes, infill_notes, total_duration,
                       context_frac=0.25, suffix_frac=0.25):
    """
    Note Density Ratio — ratio of note count in infilled vs original region.
    Value close to 1.0 means similar density; >1 = more notes, <1 = fewer.
    """
    region_dur = total_duration * (1 - context_frac - suffix_frac)
    if region_dur <= 0:
        return None
    orig_density   = len(orig_notes)   / region_dur
    infill_density = len(infill_notes) / region_dur
    if orig_density == 0:
        return None
    return float(infill_density / orig_density)


def evaluate_pair(original_path, infilled_path):
    """Evaluate a single original/infilled pair."""
    orig_notes,   resolution = load_midi_notes(original_path)
    infill_notes, _          = load_midi_notes(infilled_path)

    if orig_notes is None or infill_notes is None:
        return None

    # Total duration from last note end
    all_ends = [e for (_, _, e) in orig_notes]
    if not all_ends:
        return None
    total_duration = max(all_ends)

    if total_duration <= 0:
        return None

    # Extract middle region (middle 50% of the piece)
    orig_mid   = extract_middle_region(orig_notes,   total_duration)
    infill_mid = extract_middle_region(infill_notes, total_duration)

    if not orig_mid:
        print(f"    Warning: no notes in original middle region")
        return None

    results = {}

    p = pcho(orig_mid, infill_mid)
    if p is not None:
        results['pcho'] = round(p, 4)

    g = groove_similarity(orig_mid, infill_mid, resolution, total_duration)
    if g is not None:
        results['groove_similarity'] = round(g, 4)

    n = note_density_ratio(orig_mid, infill_mid, total_duration)
    if n is not None:
        results['note_density_ratio'] = round(n, 4)

    return results if results else None


def plot_infilling_metrics(all_results, output_path):
    """Plot infilling metrics as bar charts with error bars."""
    metrics = {
        'pcho':               ('Pitch Class\nHistogram Overlap', 'Higher = better tonal match'),
        'groove_similarity':  ('Groove Similarity',              'Higher = better rhythmic match'),
        'note_density_ratio': ('Note Density Ratio',             'Closer to 1.0 = similar density'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for ax, (key, (label, desc)) in zip(axes, metrics.items()):
        values = [r[key] for r in all_results if key in r]
        if not values:
            ax.set_title(label)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        mean = np.mean(values)
        std  = np.std(values)
        n    = len(values)

        color = '#4C72B0' if key != 'note_density_ratio' else '#DD8452'
        ax.bar([0], [mean], yerr=[std], capsize=6, color=color,
               alpha=0.85, width=0.4,
               error_kw={'linewidth': 1.5, 'capthick': 1.5})

        # Reference line
        if key == 'note_density_ratio':
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                       label='Perfect (1.0)')
            ax.legend(fontsize=8)
        elif key in ('pcho', 'groove_similarity'):
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
            ax.set_ylim(0, 1.15)

        ax.set_title(f'{label}\nn={n}', fontsize=11)
        ax.set_ylabel('Score', fontsize=10)
        ax.text(0, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_xlabel(desc, fontsize=8, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('MDLM Infilling Quality — Middle Region vs Ground Truth',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    try:
        import muspy
    except ImportError:
        print("muspy not installed. Run: pip install muspy")
        sys.exit(1)

    infill_dir = 'outputs/infilling'
    if not os.path.exists(infill_dir):
        print(f"Infilling directory not found: {infill_dir}")
        sys.exit(1)

    # Find all original/infilled pairs
    originals = sorted(glob.glob(os.path.join(infill_dir, '*_original.mid')))
    pairs = []
    for orig in originals:
        infilled = orig.replace('_original.mid', '_infilled.mid')
        if os.path.exists(infilled):
            pairs.append((orig, infilled))

    if not pairs:
        print("No original/infilled pairs found.")
        sys.exit(1)

    print(f"Evaluating {len(pairs)} original/infilled pairs...\n")

    all_results = []
    for orig, infilled in pairs:
        piece_name = os.path.basename(orig).replace('_original.mid', '')
        print(f"{piece_name}:")
        result = evaluate_pair(orig, infilled)
        if result:
            all_results.append(result)
            for k, v in result.items():
                print(f"  {k}: {v:.4f}")
        else:
            print("  Could not compute metrics")

    if not all_results:
        print("No valid results.")
        sys.exit(1)

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Metric':<30} {'Mean':>8} {'Std':>8} {'N':>5}")
    print(f"{'='*55}")
    for key in ['pcho', 'groove_similarity', 'note_density_ratio']:
        values = [r[key] for r in all_results if key in r]
        if values:
            print(f"{key:<30} {np.mean(values):>8.4f} {np.std(values):>8.4f} {len(values):>5}")
    print(f"{'='*55}")

    # Save JSON
    summary = {}
    for key in ['pcho', 'groove_similarity', 'note_density_ratio']:
        values = [r[key] for r in all_results if key in r]
        if values:
            summary[key] = {
                'mean': round(float(np.mean(values)), 4),
                'std':  round(float(np.std(values)), 4),
                'n':    len(values),
                'values': [round(v, 4) for v in values]
            }

    out_json = 'outputs/infilling_metrics.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot
    plot_infilling_metrics(all_results, 'outputs/figures/infilling_metrics.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
