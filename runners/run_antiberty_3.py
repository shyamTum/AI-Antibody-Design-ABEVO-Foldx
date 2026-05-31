import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from antiberty import AntiBERTyRunner

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
        print("Usage: python run_antiberty.py <fasta_file>")
        sys.exit(1)

    fasta_file = Path(sys.argv[1]).resolve()
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_file}")

    sequence = read_fasta(fasta_file)
    sample_name = safe_name(fasta_file.stem)

    out_dir = BASE_DIR / "results" / f"{sample_name}_antiberty"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = AntiBERTyRunner()
    raw_embeddings = runner.embed([sequence])[0].detach().cpu().numpy()

    # Trim BOS/EOS if present
    if raw_embeddings.shape[0] == len(sequence) + 2:
        residue_embeddings = raw_embeddings[1:-1]
    elif raw_embeddings.shape[0] == len(sequence):
        residue_embeddings = raw_embeddings
    else:
        raise ValueError(
            f"Unexpected embedding length {raw_embeddings.shape[0]} for sequence length {len(sequence)}"
        )

    residue_norms = np.linalg.norm(residue_embeddings, axis=1)

    residue_df = pd.DataFrame({
        "position": np.arange(1, len(sequence) + 1),
        "residue": list(sequence),
        "embedding_norm": residue_norms,
    })

    # Compact slice for lightweight table view
    dim_df = pd.DataFrame(residue_embeddings[:, :20])
    dim_df.insert(0, "residue", list(sequence))
    dim_df.insert(0, "position", np.arange(1, len(sequence) + 1))

    residue_df.to_csv(out_dir / "residue_norms.csv", index=False)
    dim_df.to_csv(out_dir / "embedding_slice.csv", index=False)

    # Save full embeddings for UMAP / richer plots
    np.save(out_dir / "residue_embeddings.npy", residue_embeddings)

    summary = {
        "tool": "AntiBERTy",
        "input_fasta": str(fasta_file),
        "sample_name": sample_name,
        "sequence_length": len(sequence),
        "raw_embedding_shape": list(raw_embeddings.shape),
        "residue_embedding_shape": list(residue_embeddings.shape),
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
