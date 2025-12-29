#!/usr/bin/env python3
"""
Extract RNA-FM embeddings for a single RNA sequence.

Usage:
    python extract_embedding.py SEQUENCE OUTPUT.npy
    python extract_embedding.py --fasta INPUT.fa OUTPUT.npy
    
Examples:
    python extract_embedding.py ACGUACGUACGU embeddings.npy
    python extract_embedding.py --fasta example/MIR921.fa mir921_emb.npy
    
Options:
    --device    Device to use: auto, cpu, mps, cuda (default: auto)
    --mean      Save mean-pooled embedding (640,) instead of per-nucleotide (L, 640)
"""

import os
import sys

# Fix macOS OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path so fm module can be found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import numpy as np
import torch
import fm


def get_device(preferred: str = "auto") -> str:
    """Get the best available device."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    elif preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("Warning: MPS not available, using CPU", file=sys.stderr)
        return "cpu"
    elif preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Warning: CUDA not available, using CPU", file=sys.stderr)
        return "cpu"
    return "cpu"


def read_fasta(fasta_path: str) -> tuple[str, str]:
    """Read first sequence from a FASTA file. Returns (name, sequence)."""
    name = None
    seq_lines = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    break  # Only read first sequence
                name = line[1:].split()[0]
            elif line:
                seq_lines.append(line.upper().replace('T', 'U'))
    
    if name is None:
        raise ValueError(f"No valid FASTA sequence found in {fasta_path}")
    
    return name, ''.join(seq_lines)


def extract_embedding(sequence: str, device: str = "auto", mean_pool: bool = False) -> np.ndarray:
    """
    Extract RNA-FM embedding for a sequence.
    
    Args:
        sequence: RNA sequence (ACGU)
        device: Device to use
        mean_pool: If True, return mean-pooled (640,), else per-nucleotide (L, 640)
    
    Returns:
        numpy array of embeddings
    """
    device = get_device(device)
    
    # Load model
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)
    
    # Prepare sequence
    sequence = sequence.upper().replace('T', 'U')
    data = [("seq", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Extract embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    
    # Get embedding and remove BOS/EOS tokens
    emb = results["representations"][12][0, 1:-1, :].cpu().numpy()
    
    if mean_pool:
        emb = emb.mean(axis=0)
    
    return emb


def main():
    parser = argparse.ArgumentParser(
        description="Extract RNA-FM embeddings for a single RNA sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ACGUACGUACGU output.npy
  %(prog)s --fasta example/MIR921.fa output.npy
  %(prog)s --mean ACGUACGU output.npy    # Mean-pooled embedding
  %(prog)s --device mps ACGU output.npy  # Force MPS on Apple Silicon
        """
    )
    parser.add_argument("sequence", help="RNA sequence (ACGU) or use --fasta for file input")
    parser.add_argument("output", help="Output path for .npy file")
    parser.add_argument("--fasta", action="store_true", 
                        help="Treat 'sequence' argument as a FASTA file path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Device to use (default: auto)")
    parser.add_argument("--mean", action="store_true",
                        help="Save mean-pooled embedding (640,) instead of per-nucleotide (L, 640)")
    
    args = parser.parse_args()
    
    # Get sequence
    if args.fasta:
        name, sequence = read_fasta(args.sequence)
        print(f"Read sequence '{name}' ({len(sequence)} nt) from {args.sequence}", file=sys.stderr)
    else:
        sequence = args.sequence
        print(f"Processing sequence ({len(sequence)} nt)", file=sys.stderr)
    
    # Extract embedding
    device = get_device(args.device)
    print(f"Using device: {device}", file=sys.stderr)
    
    emb = extract_embedding(sequence, device=args.device, mean_pool=args.mean)
    
    # Save
    np.save(args.output, emb)
    print(f"Saved embedding {emb.shape} to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

