import sys
import torch
import esm

fasta_file = sys.argv[1]

sequences = []
current_seq = []

with open(fasta_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                sequences.append("".join(current_seq))
                current_seq = []
        else:
            current_seq.append(line)

if current_seq:
    sequences.append("".join(current_seq))

if not sequences:
    raise ValueError("No sequences found in FASTA file.")

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()

for i, seq in enumerate(sequences, start=1):
    data = [(f"seq_{i}", seq)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])

    embeddings = results["representations"][6]
    print(f"Chain {i} length: {len(seq)}")
    print(f"Chain {i} embedding shape: {embeddings.shape}")
