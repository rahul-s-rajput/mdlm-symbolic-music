"""Quick sanity check: verify dataloader integration works on CPU."""
import sys
sys.path.insert(0, '.')

import datasets
import torch

print("=" * 50)
print("1. Loading vocab_info.json")
print("=" * 50)
import json
with open("data/maestro_tokenized/vocab_info.json") as f:
    info = json.load(f)
print(f"   {info}")

print("\n" + "=" * 50)
print("2. Loading train dataset from disk")
print("=" * 50)
ds = datasets.load_from_disk("data/maestro_tokenized/train").with_format("torch")
print(f"   Num chunks: {len(ds)}")
print(f"   Features: {ds.features}")

sample = ds[0]
print(f"   Sample input_ids shape: {sample['input_ids'].shape}")
print(f"   Sample attention_mask shape: {sample['attention_mask'].shape}")
print(f"   First 20 token IDs: {sample['input_ids'][:20].tolist()}")
print(f"   Last 20 token IDs: {sample['input_ids'][-20:].tolist()}")
print(f"   Min ID: {sample['input_ids'].min().item()}, Max ID: {sample['input_ids'].max().item()}")

# Check BOS/EOS placement
bos_id, eos_id, pad_id = info['bos_id'], info['eos_id'], info['pad_id']
print(f"   Starts with BOS ({bos_id})? {sample['input_ids'][0].item() == bos_id}")

# Find EOS position
eos_positions = (sample['input_ids'] == eos_id).nonzero(as_tuple=True)[0]
if len(eos_positions) > 0:
    print(f"   EOS at position: {eos_positions[0].item()}")

# Check padding
pad_count = (sample['input_ids'] == pad_id).sum().item()
print(f"   PAD tokens: {pad_count} / 1024")

print("\n" + "=" * 50)
print("3. Loading validation dataset")
print("=" * 50)
val_ds = datasets.load_from_disk("data/maestro_tokenized/validation").with_format("torch")
print(f"   Num chunks: {len(val_ds)}")

print("\n" + "=" * 50)
print("4. Loading test dataset")
print("=" * 50)
test_ds = datasets.load_from_disk("data/maestro_tokenized/test").with_format("torch")
print(f"   Num chunks: {len(test_ds)}")

print("\n" + "=" * 50)
print("5. Testing MidiTokenizer wrapper")
print("=" * 50)
from dataloader import MidiTokenizer
tok = MidiTokenizer(data_dir="data/maestro_tokenized")
print(f"   vocab_size: {tok.vocab_size}")
print(f"   bos_token_id: {tok.bos_token_id}")
print(f"   eos_token_id: {tok.eos_token_id}")
print(f"   mask_token_id: {tok.mask_token_id}")
print(f"   mask_token: {tok.mask_token}")
print(f"   pad_token_id: {tok.pad_token_id}")
print(f"   len(tokenizer): {len(tok)}")

# Test decode
decoded = tok.decode(sample['input_ids'][:10])
print(f"   decode(first 10): '{decoded}'")

print("\n" + "=" * 50)
print("6. Testing DataLoader batch")
print("=" * 50)
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
print(f"   Batch attention_mask shape: {batch['attention_mask'].shape}")
print(f"   dtype: {batch['input_ids'].dtype}")

# Verify all IDs are within vocab range
max_id = batch['input_ids'].max().item()
print(f"   Max token ID in batch: {max_id}")
print(f"   Vocab size: {tok.vocab_size}")
print(f"   All IDs < vocab_size? {max_id < tok.vocab_size}")

print("\n" + "=" * 50)
print("ALL CHECKS PASSED" if max_id < tok.vocab_size else "WARNING: IDs out of range!")
print("=" * 50)
