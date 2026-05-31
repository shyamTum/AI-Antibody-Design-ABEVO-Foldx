import sys
import os
import pandas as pd

# Path to efficient-evolution repo
EE_PATH = os.path.expanduser("~/Desktop/Shyam/tools/efficient-evolution")
sys.path.insert(0, EE_PATH)
sys.path.insert(0, os.path.join(EE_PATH, "bin"))

from predict_esm import predict_esm  # noqa: E402

def read_fasta(path: str) -> str:
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)

if len(sys.argv) < 2:
    print("Usage: python run_abevo.py <fasta_file>")
    sys.exit(1)

fasta_file = sys.argv[1]
sequence = read_fasta(fasta_file)

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

print("Sequence length:", len(sequence))
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(10).to_string(index=False))
