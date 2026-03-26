"""
generate_samples.py — Unconditional MIDI generation from AR and MDLM checkpoints.

Usage (from duo/ directory):
    python generate_samples.py --model ar --checkpoint ../ar_best.ckpt --num_samples 8
    python generate_samples.py --model mdlm --checkpoint ../mdlm_best_nll1.741.ckpt --num_samples 8

Outputs:
    outputs/generated/ar/sample_0.mid ... sample_7.mid
    outputs/generated/mdlm/sample_0.mid ... sample_7.mid
"""

import argparse
import os
import sys

import torch
import numpy as np

# Add duo/ to path so we can import from the codebase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import algo
import dataloader


def load_model(model_type, checkpoint_path, data_dir, device):
    """Load a trained AR or MDLM model from checkpoint."""
    print(f"Loading {model_type.upper()} from {checkpoint_path}...")

    # Patch torch.load for PyTorch 2.6 compatibility
    import omegaconf
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    # Load tokenizer
    tokenizer = dataloader.MidiTokenizer(data_dir=data_dir)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, "
          f"MASK={tokenizer.mask_token_id}, PAD={tokenizer.pad_token_id}")

    # Load model class
    if model_type == 'ar':
        model_cls = algo.AR
    elif model_type == 'mdlm':
        model_cls = algo.MDLM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load from checkpoint
    model = model_cls.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    # Use EMA weights for generation (best quality)
    if model.ema is not None:
        print("  Using EMA weights for generation")
        model.ema.store(model._get_parameters())
        model.ema.copy_to(model._get_parameters())

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer


def tokens_to_midi(token_ids, tokenizer_path, output_path):
    """Convert a sequence of token IDs back to a MIDI file using MidiTok v3."""
    import json

    # Load vocab info
    with open(os.path.join(tokenizer_path, 'vocab_info.json')) as f:
        info = json.load(f)
    special_ids = {info['pad_id'], info['bos_id'],
                   info['eos_id'], info['mask_id']}

    # Remove special tokens
    clean_ids = [t for t in token_ids
                 if t not in special_ids and t < info['vocab_size']]

    if len(clean_ids) < 10:
        print(f"    Warning: only {len(clean_ids)} tokens after filtering, skipping")
        return False

    try:
        from miditok import REMI
        tokenizer = REMI(params=os.path.join(tokenizer_path, 'tokenizer.json'))
        score = tokenizer([clean_ids])
        score.dump_midi(output_path)
        print(f"    Saved: {output_path} ({len(clean_ids)} tokens)")
        return True

    except Exception as e:
        print(f"    MIDI conversion error: {e}")
        np.save(output_path.replace('.mid', '_tokens.npy'), np.array(token_ids))
        print(f"    Saved raw tokens: {output_path.replace('.mid', '_tokens.npy')}")
        return False


def generate(model_type, checkpoint_path, data_dir, output_dir,
             num_samples=8, sampling_steps=1000, device='cuda'):
    """Generate samples and save as MIDI files."""

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(
        model_type, checkpoint_path, data_dir, device)

    print(f"\nGenerating {num_samples} samples...")
    print(f"  Sampling steps: {sampling_steps}")
    print(f"  Sequence length: {model.num_tokens}")

    # Generate in batches to avoid OOM on small GPUs (e.g. GTX 1650 4GB)
    batch_size = 4  # safe for 4GB VRAM; increase if you have more
    all_samples = []

    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            this_batch = min(batch_size, num_samples - batch_start)
            print(f"  Batch {batch_start // batch_size + 1}: generating {this_batch} samples...")
            if model_type == 'ar':
                batch = model.generate_samples(num_samples=this_batch)
            elif model_type == 'mdlm':
                batch = model.generate_samples(
                    num_samples=this_batch,
                    num_steps=sampling_steps)
            all_samples.append(batch.cpu())
            torch.cuda.empty_cache()

    samples = torch.cat(all_samples, dim=0)
    print(f"  Generated tensor shape: {samples.shape}")

    # Decode and save each sample
    # Find next available index to avoid overwriting existing samples
    existing = [f for f in os.listdir(output_dir) if f.startswith('sample_') and f.endswith('.mid')]
    start_idx = max([int(f.replace('sample_','').replace('.mid','')) for f in existing], default=-1) + 1
    if start_idx > 0:
        print(f"  Found {start_idx} existing samples, starting from sample_{start_idx}.mid")

    print(f"\nSaving MIDI files to {output_dir}/")
    success_count = 0
    for i, sample in enumerate(samples):
        token_ids = sample.cpu().tolist()
        output_path = os.path.join(output_dir, f"sample_{start_idx + i}.mid")
        success = tokens_to_midi(token_ids, data_dir, output_path)
        if success:
            success_count += 1

    print(f"\nDone: {success_count}/{num_samples} samples saved as MIDI")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate MIDI samples from AR or MDLM checkpoint')
    parser.add_argument('--model', type=str, required=True,
                        choices=['ar', 'mdlm'],
                        help='Model type: ar or mdlm')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .ckpt file')
    parser.add_argument('--data_dir', type=str,
                        default='data/maestro_tokenized',
                        help='Path to tokenizer/vocab_info.json directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/generated/<model>)')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to generate')
    parser.add_argument('--sampling_steps', type=int, default=1000,
                        help='Number of denoising steps for MDLM')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(
            'outputs', 'generated', args.model)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    print(f"Using device: {args.device}")

    # Run generation
    generate(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        device=args.device,
    )


if __name__ == '__main__':
    main()
