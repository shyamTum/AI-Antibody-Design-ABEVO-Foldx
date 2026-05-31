import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_esm2.py <fasta_file>")
        sys.exit(1)

    fasta_file = Path(sys.argv[1]).resolve()
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_file}")

    sequence = read_fasta(fasta_file)
    sample_name = safe_name(fasta_file.stem)

    out_dir = BASE_DIR / "results" / f"{sample_name}_esm2"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [(sample_name, sequence)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])

    token_embeddings = results["representations"][6][0].cpu().numpy()
    residue_embeddings = token_embeddings[1:len(sequence) + 1]

    residue_norms = np.linalg.norm(residue_embeddings, axis=1)

    residue_df = pd.DataFrame({
        "position": np.arange(1, len(sequence) + 1),
        "residue": list(sequence),
        "embedding_norm": residue_norms,
    })

    dim_df = pd.DataFrame(residue_embeddings[:, :20])
    dim_df.insert(0, "residue", list(sequence))
    dim_df.insert(0, "position", np.arange(1, len(sequence) + 1))

    residue_df.to_csv(out_dir / "residue_norms.csv", index=False)
    dim_df.to_csv(out_dir / "embedding_slice.csv", index=False)
    np.save(out_dir / "residue_embeddings.npy", residue_embeddings)

    summary = {
        "tool": "ESM-2",
        "input_fasta": str(fasta_file),
        "sample_name": sample_name,
        "sequence_length": len(sequence),
        "embedding_shape": list(residue_embeddings.shape),
        "results_dir": str(out_dir),
        "residue_norms_csv": str(out_dir / "residue_norms.csv"),
        "embedding_slice_csv": str(out_dir / "embedding_slice.csv"),
        "residue_embeddings_npy": str(out_dir / "residue_embeddings.npy"),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nResidue norms preview:")
    print(residue_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
