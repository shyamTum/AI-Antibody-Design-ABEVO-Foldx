import os
import re
import sys
import json
from pathlib import Path

import pandas as pd

# ---------- paths ----------
BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
EE_PATH = Path("/home/ghoshlab/Desktop/Shyam/tools/efficient-evolution")

sys.path.insert(0, str(EE_PATH))
sys.path.insert(0, str(EE_PATH / "bin"))

from predict_esm import predict_esm  # noqa: E402


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


def aa_position(mut: str) -> int:
    """
    Works for strings like D0A, A114V, etc.
    Extracts numeric position.
    """
    m = re.search(r"(\d+)", mut)
    if not m:
        return -1
    return int(m.group(1))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_abevo.py <fasta_file>")
        sys.exit(1)

    fasta_file = Path(sys.argv[1]).resolve()
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_file}")

    sequence = read_fasta(fasta_file)
    sample_name = safe_name(fasta_file.stem)

    out_dir = BASE_DIR / "results" / f"{sample_name}_abevo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start with 2 models only for CPU friendliness
    model_locations = [
        "esm1b_t33_650M_UR50S",
        "esm1v_t33_650M_UR90S_1",
    ]

    df = predict_esm(
        sequence,
        model_locations=model_locations,
        scoring_strategy="wt-marginals",
        mutation_col="mutant",
        offset_idx=0,
        nogpu=True,
        verbose=0,
    )

    # Combined score
    df["score_mean"] = df[model_locations].mean(axis=1)
    df["position"] = df["mutant"].apply(aa_position)

    # Position-level summary
    pos_df = (
        df.groupby("position", as_index=False)["score_mean"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "position_mean_score"})
    )

    # Top / bottom mutations
    top_df = df.sort_values("score_mean", ascending=False).head(20).copy()
    bottom_df = df.sort_values("score_mean", ascending=True).head(20).copy()

    # Save outputs
    df.to_csv(out_dir / "mutations.csv", index=False)
    pos_df.to_csv(out_dir / "position_scores.csv", index=False)
    top_df.to_csv(out_dir / "top_mutations.csv", index=False)
    bottom_df.to_csv(out_dir / "bottom_mutations.csv", index=False)

    summary = {
        "tool": "Antibody-Evolution",
        "input_fasta": str(fasta_file),
        "sample_name": sample_name,
        "sequence_length": len(sequence),
        "num_mutations": int(len(df)),
        "models_used": model_locations,
        "results_dir": str(out_dir),
        "mutations_csv": str(out_dir / "mutations.csv"),
        "position_scores_csv": str(out_dir / "position_scores.csv"),
        "top_mutations_csv": str(out_dir / "top_mutations.csv"),
        "bottom_mutations_csv": str(out_dir / "bottom_mutations.csv"),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nTop 10 mutations:")
    print(top_df[["mutant", *model_locations, "score_mean"]].head(10).to_string(index=False))
    print("\nBottom 10 mutations:")
    print(bottom_df[["mutant", *model_locations, "score_mean"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
