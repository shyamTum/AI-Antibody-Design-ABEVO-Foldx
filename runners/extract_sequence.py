from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import sys

pdb_file = sys.argv[1]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_file)

sequences = {}

for model in structure:
    for chain in model:
        seq = ""
        for residue in chain:
            if residue.get_resname() != "HOH":
                try:
                    seq += seq1(residue.get_resname())
                except:
                    continue
        if seq:
            sequences[chain.id] = seq

for chain, seq in sequences.items():
    print(f">{pdb_file}_chain_{chain}")
    print(seq)
