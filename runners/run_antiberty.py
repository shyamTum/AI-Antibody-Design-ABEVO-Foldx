import sys
from antiberty import AntiBERTyRunner

fasta_file = sys.argv[1]

# read sequence
with open(fasta_file, "r") as f:
    lines = f.readlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])

runner = AntiBERTyRunner()

embeddings = runner.embed([seq])

print("Sequence length:", len(seq))
print("Embedding shape:", embeddings[0].shape)
