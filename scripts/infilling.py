"""
infilling.py — MDLM infilling experiment.

Takes real MAESTRO validation chunks, masks the middle portion,
and lets MDLM fill it in. AR cannot do this.

Usage (from duo/ directory):
    python infilling.py --checkpoint ../mdlm_best_nll1.741.ckpt --num_pieces 4

Outputs:
    outputs/infilling/piece_0_original.mid
    outputs/infilling/piece_0_infilled.mid
    ... (for each piece)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import algo
import dataloader
from generate_samples import load_model, tokens_to_midi


def load_validation_chunks(data_dir, num_pieces=4, seed=42):
    """Load real MAESTRO validation chunks to use as infilling targets."""
    val_path = os.path.join(data_dir, 'validation')
    print(f"Loading validation data from {val_path}...")
    ds = datasets.load_from_disk(val_path).with_format('torch')
    print(f"  {len(ds)} validation chunks available")

    # Pick deterministically using seed
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=num_pieces, replace=False)
    indices = sorted(indices.tolist())
    print(f"  Selected chunk indices: {indices}")

    chunks = []
    for idx in indices:
        chunk = ds[idx]['input_ids']
        chunks.append(chunk)
    return chunks


def create_infilling_input(tokens, mask_token_id,
                           context_len=256, suffix_len=256):
    """
    Mask the middle of a token sequence for infilling.

    Layout: [context_len tokens] [MASK...MASK] [suffix_len tokens]
    Middle region length = total_len - context_len - suffix_len

    Returns:
        masked_tokens: tensor with middle replaced by MASK
        mask_region: (start, end) indices of the masked region
    """
    total_len = len(tokens)
    middle_start = context_len
    middle_end = total_len - suffix_len
    middle_len = middle_end - middle_start

    assert middle_len > 0, \
        f"Sequence too short: {total_len} <= {context_len + suffix_len}"

    masked = tokens.clone()
    masked[middle_start:middle_end] = mask_token_id

    print(f"  Sequence length: {total_len}")
    print(f"  Context: tokens 0-{middle_start} ({context_len} tokens)")
    print(f"  Masked:  tokens {middle_start}-{middle_end} ({middle_len} tokens)")
    print(f"  Suffix:  tokens {middle_end}-{total_len} ({suffix_len} tokens)")

    return masked, (middle_start, middle_end)


@torch.no_grad()
def infill(model, masked_tokens, mask_region, num_steps=500, device='cuda'):
    """
    Run MDLM denoising conditioned on the known tokens.

    Key insight: at each denoising step, after sampling the next x,
    we clamp back the known (non-masked) tokens to their original values.
    This is the standard 'replacement' trick for conditional generation
    with diffusion models.
    """
    mask_start, mask_end = mask_region
    mask_token_id = model.mask_index
    
    x = masked_tokens.unsqueeze(0).to(device)  # (1, seq_len)
    known_mask = torch.ones(x.shape, dtype=torch.bool, device=device)
    known_mask[:, mask_start:mask_end] = False  # False = region to infill

    # Time schedule
    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

    p_x0_cache = None
    for i in range(num_steps):
        t = timesteps[i] * torch.ones(1, 1, device=device)
        dt = (timesteps[i] - timesteps[i + 1]).item()

        p_x0_cache, x_next = model._ancestral_update(
            x=x, t=t, labels=None, dt=dt, p_x0=p_x0_cache)

        # Cache invalidation
        if (not torch.allclose(x_next, x) or model.time_conditioning):
            p_x0_cache = None

        x = x_next

        # Clamp known tokens back — this is the conditional infilling trick
        x[known_mask] = masked_tokens.to(device)[known_mask.squeeze(0)]

    # Final noise removal step
    t0 = timesteps[-1] * torch.ones(1, 1, device=device)
    _, x = model._ancestral_update(
        x=x, t=t0, labels=None, dt=None,
        p_x0=p_x0_cache, noise_removal_step=True)

    # Clamp one final time
    x[known_mask] = masked_tokens.to(device)[known_mask.squeeze(0)]

    return x.squeeze(0)


def main():
    parser = argparse.ArgumentParser(
        description='MDLM infilling experiment on MAESTRO validation chunks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MDLM .ckpt file')
    parser.add_argument('--data_dir', type=str,
                        default='data/maestro_tokenized',
                        help='Path to tokenized MAESTRO data')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/infilling',
                        help='Output directory for MIDI files')
    parser.add_argument('--num_pieces', type=int, default=4,
                        help='Number of pieces to infill')
    parser.add_argument('--context_len', type=int, default=256,
                        help='Tokens to keep at the start (context)')
    parser.add_argument('--suffix_len', type=int, default=256,
                        help='Tokens to keep at the end (suffix)')
    parser.add_argument('--num_steps', type=int, default=500,
                        help='Number of MDLM denoising steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for chunk selection')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MDLM model
    model, tokenizer = load_model(
        'mdlm', args.checkpoint, args.data_dir, args.device)

    # Load validation chunks
    chunks = load_validation_chunks(
        args.data_dir, args.num_pieces, args.seed)

    print(f"\nRunning infilling on {len(chunks)} pieces...")
    print(f"  Context: {args.context_len} tokens")
    print(f"  Suffix:  {args.suffix_len} tokens")
    print(f"  Denoising steps: {args.num_steps}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Piece {i} ---")

        original_path = os.path.join(args.output_dir, f'piece_{i}_original.mid')
        masked_path   = os.path.join(args.output_dir, f'piece_{i}_masked.mid')
        infilled_path = os.path.join(args.output_dir, f'piece_{i}_infilled.mid')

        # Skip if all three files already exist
        if all(os.path.exists(p) for p in [original_path, masked_path, infilled_path]):
            print(f"  Skipping piece {i} — files already exist")
            continue

        # Save original MIDI
        tokens_to_midi(chunk.tolist(), args.data_dir, original_path)

        # Create masked version
        masked_tokens, mask_region = create_infilling_input(
            chunk, tokenizer.mask_token_id,
            args.context_len, args.suffix_len)

        tokens_to_midi(masked_tokens.tolist(), args.data_dir, masked_path)

        # Run infilling
        print(f"  Running MDLM infilling ({args.num_steps} steps)...")
        infilled_tokens = infill(
            model, masked_tokens, mask_region,
            num_steps=args.num_steps, device=args.device)

        tokens_to_midi(infilled_tokens.tolist(), args.data_dir, infilled_path)

        print(f"  Saved: original, masked, infilled for piece {i}")

    print(f"\nDone! Files saved to {args.output_dir}/")
    print("\nFor each piece you have:")
    print("  piece_N_original.mid — real MAESTRO music")
    print("  piece_N_masked.mid   — with middle silenced (what AR would see)")
    print("  piece_N_infilled.mid — MDLM's infilled version")


if __name__ == '__main__':
    main()
