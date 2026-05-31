from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import sys

pdb_file = sys.argv[1]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_file)

base = os.path.splitext(os.path.basename(pdb_file))[0]
out_dir = os.path.dirname(pdb_file)

for model in structure:
    for chain in model:
        seq = []
        seen = set()
        for residue in chain:
            rid = residue.id
            if rid in seen:
                continue
            seen.add(rid)
            if residue.id[0] != " ":
                continue
            try:
                aa = seq1(residue.get_resname())
                seq.append(aa)
            except Exception:
                continue

        if seq:
            chain_seq = "".join(seq)
            out_file = os.path.join(out_dir, f"{base}_chain_{chain.id}.fasta")
            with open(out_file, "w") as f:
                f.write(f">{base}_chain_{chain.id}\n")
                f.write(chain_seq + "\n")
            print(f"Wrote {out_file} length={len(chain_seq)}")
