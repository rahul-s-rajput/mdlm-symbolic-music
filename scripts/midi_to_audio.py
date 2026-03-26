"""
midi_to_audio.py — Convert all generated MIDI files to WAV.

Uses tinysoundfont — self-contained Python package, no external DLL needed.

Install:
    pip install tinysoundfont mido

Usage (from duo/ directory):
    python midi_to_audio.py --soundfont path/to/soundfont.sf2
"""

import argparse
import glob
import os
import sys
import wave
import numpy as np


def find_midi_files(root_dirs):
    midi_files = []
    for d in root_dirs:
        if os.path.exists(d):
            midi_files.extend(
                glob.glob(os.path.join(d, '**', '*.mid'), recursive=True))
    return sorted(midi_files)


def convert_midi_to_wav(midi_path, wav_path, soundfont, sample_rate=44100):
    """
    Convert a MIDI file to WAV using tinysoundfont offline rendering.

    Key fix: tinysoundfont generate() can return values outside [-1, 1]
    when many notes play simultaneously (common for piano). We normalize
    by peak amplitude instead of hard-clipping, which prevents noise.
    """
    import tinysoundfont

    # gain=-20 dB prevents clipping from many simultaneous piano notes
    synth = tinysoundfont.Synth(samplerate=int(sample_rate), gain=-20)
    sfid = synth.sfload(soundfont)
    if sfid < 0:
        raise RuntimeError(f"Failed to load soundfont: {soundfont}")

    # Attach sequencer BEFORE calling generate()
    # generate() drives the sequencer internally — do NOT call seq.process() separately
    seq = tinysoundfont.Sequencer(synth)
    seq.midi_load(midi_path)

    # Offline rendering: just call generate() in a loop
    # The sequencer callback fires automatically inside generate() at correct sample offsets
    chunk_samples = int(0.05 * sample_rate)  # 50ms chunks
    all_chunks = []

    while not seq.is_empty():
        raw = synth.generate(chunk_samples)
        # generate() returns a memoryview of stereo interleaved float32 samples
        chunk = np.frombuffer(raw, dtype=np.float32).copy()
        all_chunks.append(chunk)

    # 3s tail for sustain/reverb decay
    for _ in range(int(3.0 / 0.05)):
        raw = synth.generate(chunk_samples)
        all_chunks.append(np.frombuffer(raw, dtype=np.float32).copy())

    if not all_chunks:
        raise RuntimeError("No audio generated — is the MIDI file valid?")

    audio = np.concatenate(all_chunks)  # interleaved stereo float32 in [-1, 1]

    peak = float(np.abs(audio).max())
    rms  = float(np.sqrt(np.mean(audio ** 2)))

    if peak == 0:
        raise RuntimeError("Audio is silent — soundfont may not have loaded")

    # Normalize to 0.95 peak to avoid any remaining clipping
    audio = audio / peak * 0.95

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write stereo WAV
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    duration_s = len(audio_int16) / 2 / sample_rate
    size_kb    = os.path.getsize(wav_path) / 1024
    print(f"  ✓ {os.path.basename(wav_path)} "
          f"({duration_s:.1f}s, {size_kb:.0f} KB, peak={peak:.3f}, rms={rms:.4f})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert generated MIDI files to WAV using tinysoundfont')
    parser.add_argument('--soundfont', type=str, required=True,
                        help='Path to .sf2 soundfont file')
    parser.add_argument('--dirs', nargs='+',
                        default=['outputs/generated', 'outputs/infilling'],
                        help='Directories to search for .mid files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing .wav files')
    parser.add_argument('--sample_rate', type=int, default=44100)
    args = parser.parse_args()

    try:
        import tinysoundfont
    except ImportError:
        print("tinysoundfont not installed. Run: pip install tinysoundfont")
        sys.exit(1)

    if not os.path.exists(args.soundfont):
        print(f"ERROR: Soundfont not found: {args.soundfont}")
        sys.exit(1)

    midi_files = find_midi_files(args.dirs)
    if not midi_files:
        print(f"No .mid files found in: {args.dirs}")
        sys.exit(1)

    if not args.overwrite:
        midi_files = [f for f in midi_files
                      if not os.path.exists(f.replace('.mid', '.wav'))]

    if not midi_files:
        print("All files already converted. Use --overwrite to redo.")
        sys.exit(0)

    print(f"Converting {len(midi_files)} MIDI files to WAV...")
    print(f"Soundfont: {args.soundfont}\n")

    success = 0
    for f in midi_files:
        wav_path = f.replace('.mid', '.wav')
        try:
            convert_midi_to_wav(f, wav_path, args.soundfont, args.sample_rate)
            success += 1
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)} — {e}")

    print(f"\nDone: {success}/{len(midi_files)} converted successfully")


if __name__ == '__main__':
    main()
