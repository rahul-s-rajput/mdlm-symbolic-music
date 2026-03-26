"""
Preprocess MAESTRO v3 MIDI files into tokenized chunks for DUO training.

This script:
1. Tokenizes all MIDI files using MidiTok REMI
2. Splits into fixed-length chunks (1024 tokens, 512 overlap)
3. Saves train/val/test splits as HuggingFace Datasets on disk

Usage:
    python preprocess_maestro.py --maestro_dir /path/to/maestro-v3.0.0 --output_dir ./data/maestro_tokenized
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import datasets
import torch
from miditok import REMI, TokenizerConfig
from tqdm import tqdm


def get_token_ids(tokens):
    """Extract token IDs from MidiTok output.
    
    MidiTok returns a list of TokSequence objects.
    For single-track MIDI (like MAESTRO piano), we want tokens[0].ids
    """
    if isinstance(tokens, list):
        if len(tokens) > 0 and hasattr(tokens[0], 'ids'):
            return tokens[0].ids
        return tokens
    if hasattr(tokens, 'ids'):
        return tokens.ids
    return tokens


def create_tokenizer(save_dir=None):
    """Create and optionally save a MidiTok REMI tokenizer.
    
    Config matches our earlier exploration:
    - pitch_range=(21, 109) for full piano range
    - num_velocities=32
    - special_tokens includes MASK for MDLM/DUO
    """
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=False,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
    )
    tokenizer = REMI(config)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(Path(save_dir) / "tokenizer.json")
        print(f"Tokenizer saved to {save_dir}/tokenizer.json")
    
    return tokenizer


def load_maestro_splits(maestro_dir):
    """Load MAESTRO v3 metadata and return file paths per split."""
    maestro_dir = Path(maestro_dir)
    
    # MAESTRO v3 has a JSON metadata file
    json_path = maestro_dir / "maestro-v3.0.0.json"
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
        
        splits = {"train": [], "validation": [], "test": []}
        for key in metadata.get("midi_filename", {}).keys():
            split = metadata["split"][key]
            midi_file = metadata["midi_filename"][key]
            midi_path = maestro_dir / midi_file
            if midi_path.exists():
                splits[split].append(str(midi_path))
        return splits
    
    elif csv_path.exists():
        import csv
        splits = {"train": [], "validation": [], "test": []}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                midi_file = row["midi_filename"]
                midi_path = maestro_dir / midi_file
                if midi_path.exists():
                    splits[split].append(str(midi_path))
        return splits
    
    else:
        raise FileNotFoundError(
            f"Could not find maestro-v3.0.0.json or .csv in {maestro_dir}. "
            "Make sure you've extracted the MAESTRO v3 dataset."
        )


def tokenize_and_chunk(midi_paths, tokenizer, seq_len=1024, 
                       overlap=512, augment_pitches=None):
    """Tokenize MIDI files and split into fixed-length chunks.
    
    Args:
        midi_paths: List of paths to MIDI files
        tokenizer: MidiTok tokenizer
        seq_len: Target sequence length (default 1024)
        overlap: Overlap between consecutive chunks (default 512)
        augment_pitches: List of semitone shifts for data augmentation.
                        e.g., range(-5, 6) for +/-5 semitones (11x augmentation).
                        None = no augmentation.
    Returns:
        List of token ID lists, each of length seq_len
    """
    # MidiTok 3.x: access special token IDs via tokenizer["TOKEN_None"]
    bos_id = tokenizer["BOS_None"]
    eos_id = tokenizer["EOS_None"]
    pad_id = tokenizer["PAD_None"]
    
    stride = seq_len - overlap
    all_chunks = []
    skipped = 0
    
    pitch_shifts = augment_pitches if augment_pitches else [0]
    
    for midi_path in tqdm(midi_paths, desc="Tokenizing"):
        for shift in pitch_shifts:
            try:
                tokens = tokenizer(Path(midi_path))
                ids = get_token_ids(tokens)
            except Exception as e:
                skipped += 1
                continue
            
            if len(ids) < 10:
                skipped += 1
                continue
            
            # Apply pitch shift by modifying token IDs
            if shift != 0:
                ids = _apply_pitch_shift(ids, tokenizer, shift)
                if ids is None:
                    continue
            
            # Chunk with overlap using wrap=False approach:
            # [BOS] + chunk_tokens + [EOS], padded to seq_len
            content_len = seq_len - 2  # Reserve space for BOS and EOS
            
            for start in range(0, len(ids), stride):
                chunk_tokens = ids[start:start + content_len]
                
                # Skip very short final chunks (< 25% of content length)
                if len(chunk_tokens) < content_len // 4:
                    continue
                
                # Build: [BOS] + tokens + [EOS] + [PAD...]
                chunk = [bos_id] + chunk_tokens + [eos_id]
                
                # Pad to seq_len
                if len(chunk) < seq_len:
                    chunk = chunk + [pad_id] * (seq_len - len(chunk))
                
                chunk = chunk[:seq_len]
                all_chunks.append(chunk)
    
    if skipped > 0:
        print(f"  Skipped {skipped} files/augmentations (errors or too short)")
    
    return all_chunks


def _apply_pitch_shift(ids, tokenizer, semitones):
    """Shift pitch tokens by a number of semitones.
    
    Returns None if any pitch goes out of range.
    """
    vocab = tokenizer.vocab
    
    # Build mapping: token_id -> (pitch_value, token_name)
    pitch_tokens = {}
    for token_name, token_id in vocab.items():
        if token_name.startswith("Pitch_"):
            try:
                pitch_val = int(token_name.split("_")[1])
                pitch_tokens[token_id] = (pitch_val, token_name)
            except (ValueError, IndexError):
                continue
    
    if not pitch_tokens:
        return ids
    
    min_pitch = min(p for p, _ in pitch_tokens.values())
    max_pitch = max(p for p, _ in pitch_tokens.values())
    
    shifted_ids = []
    for tid in ids:
        if tid in pitch_tokens:
            old_pitch, _ = pitch_tokens[tid]
            new_pitch = old_pitch + semitones
            
            if new_pitch < min_pitch or new_pitch > max_pitch:
                return None
            
            new_token_name = f"Pitch_{new_pitch}"
            if new_token_name in vocab:
                shifted_ids.append(vocab[new_token_name])
            else:
                return None
        else:
            shifted_ids.append(tid)
    
    return shifted_ids


def save_as_hf_dataset(chunks, output_path, pad_id=0):
    """Save tokenized chunks as a HuggingFace Dataset on disk."""
    input_ids = torch.tensor(chunks, dtype=torch.long)
    attention_masks = (input_ids != pad_id).long()
    
    ds = datasets.Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    })
    ds.set_format(type="torch")
    ds.save_to_disk(output_path)
    print(f"  Saved {len(ds)} chunks to {output_path}")
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MAESTRO v3 for DUO training")
    parser.add_argument(
        "--maestro_dir", type=str, required=True,
        help="Path to extracted MAESTRO v3 directory")
    parser.add_argument(
        "--output_dir", type=str, default="./data/maestro_tokenized",
        help="Where to save tokenized datasets")
    parser.add_argument(
        "--seq_len", type=int, default=1024,
        help="Sequence length for chunks")
    parser.add_argument(
        "--overlap", type=int, default=512,
        help="Overlap between chunks")
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply pitch augmentation (training data only)")
    parser.add_argument(
        "--augment_range", type=int, default=5,
        help="Semitone range for augmentation (default +/-5 = 11x)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create tokenizer
    print("=" * 60)
    print("Step 1: Creating MidiTok REMI tokenizer")
    print("=" * 60)
    tokenizer = create_tokenizer(save_dir=str(output_dir))
    
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")
    
    # MidiTok 3.x: special token IDs are accessed via tokenizer["TOKEN_None"]
    # or from tokenizer.vocab which maps token strings to IDs
    pad_id = tokenizer["PAD_None"]
    bos_id = tokenizer["BOS_None"]
    eos_id = tokenizer["EOS_None"]
    mask_id = tokenizer["MASK_None"]
    print(f"  Special tokens: PAD={pad_id}, BOS={bos_id}, EOS={eos_id}, MASK={mask_id}")
    
    # Save vocab info for DUO config
    vocab_info = {
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "mask_id": mask_id,
        "seq_len": args.seq_len,
    }
    with open(output_dir / "vocab_info.json", "w") as f:
        json.dump(vocab_info, f, indent=2)
    print(f"  Vocab info saved to {output_dir / 'vocab_info.json'}")
    
    # Step 2: Load MAESTRO splits
    print("\n" + "=" * 60)
    print("Step 2: Loading MAESTRO v3 splits")
    print("=" * 60)
    splits = load_maestro_splits(args.maestro_dir)
    for split_name, paths in splits.items():
        print(f"  {split_name}: {len(paths)} MIDI files")
    
    # Step 3: Tokenize and chunk each split
    print("\n" + "=" * 60)
    print("Step 3: Tokenizing and chunking")
    print("=" * 60)
    
    augment_pitches = None
    if args.augment:
        augment_pitches = list(range(-args.augment_range, 
                                     args.augment_range + 1))
        print(f"  Pitch augmentation: {augment_pitches} "
              f"({len(augment_pitches)}x data)")
    
    for split_name, paths in splits.items():
        print(f"\n--- {split_name} ---")
        
        # Only augment training data
        aug = augment_pitches if split_name == "train" else None
        
        chunks = tokenize_and_chunk(
            paths, tokenizer,
            seq_len=args.seq_len,
            overlap=args.overlap,
            augment_pitches=aug,
        )
        
        print(f"  Total chunks: {len(chunks)}")
        
        if len(chunks) > 0:
            save_path = str(output_dir / split_name)
            save_as_hf_dataset(chunks, save_path, pad_id=pad_id)
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Done! Summary:")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Overlap: {args.overlap}")
    print(f"\nNext steps:")
    print(f"  1. Upload {output_dir} to Kaggle as a dataset")
    print(f"  2. Run smoke test on Kaggle with GPU T4x2")


if __name__ == "__main__":
    main()
