import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
FOLDX_DIR = Path("/home/ghoshlab/Desktop/Shyam/tools/foldx/foldx_Linux")
FOLDX_BIN = FOLDX_DIR / "foldx_20261231"
ROTABASE = FOLDX_DIR / "rotabase.txt"


ENERGY_KEYS = [
    "BackHbond",
    "SideHbond",
    "Energy_VdW",
    "Electro",
    "Energy_SolvP",
    "Energy_SolvH",
    "Energy_vdwclash",
    "energy_torsion",
    "backbone_vdwclash",
    "Entropy_sidec",
    "Entropy_mainc",
    "water_bonds",
    "helix_dipole",
    "loop_entropy",
    "cis_bond",
    "disulfide",
    "kn electrostatic",
    "partial covalent interactions",
    "Energy_Ionisation",
    "Entropy Complex",
    "Total",
]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_energy_from_log(log_text: str) -> pd.DataFrame:
    rows = []
    for line in log_text.splitlines():
        if "=" not in line:
            continue
        parts = line.split("=")
        if len(parts) < 2:
            continue

        key = parts[0].strip()
        value_str = parts[1].strip()

        # normalize key spacing
        key = re.sub(r"\s+", " ", key)

        # try parse numeric value
        try:
            value = float(value_str)
        except ValueError:
            continue

        rows.append({"term": key, "value": value})

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["term"])


def run_cmd(cmd, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python run_foldx.py <pdb_file> <mutation>")
        print("Example mutation: AH114V")
        sys.exit(1)

    pdb_file = Path(sys.argv[1]).resolve()
    mutation = sys.argv[2].strip().rstrip(";") + ";"

    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_file}")
    if not FOLDX_BIN.exists():
        raise FileNotFoundError(f"FoldX binary not found: {FOLDX_BIN}")
    if not ROTABASE.exists():
        raise FileNotFoundError(f"rotabase.txt not found: {ROTABASE}")

    sample_name = safe_name(pdb_file.stem)
    mutation_name = safe_name(mutation.replace(";", ""))
    out_dir = BASE_DIR / "results" / f"{sample_name}_foldx_{mutation_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy required files into isolated run dir
    local_pdb = out_dir / pdb_file.name
    shutil.copy2(pdb_file, local_pdb)
    shutil.copy2(ROTABASE, out_dir / "rotabase.txt")

    # Step 1: RepairPDB
    repair_cmd = [
        str(FOLDX_BIN),
        "--command=RepairPDB",
        f"--pdb={local_pdb.name}",
    ]
    repair_res = run_cmd(repair_cmd, out_dir)

    repair_stdout = repair_res.stdout
    repair_stderr = repair_res.stderr

    repaired_pdb = out_dir / f"{pdb_file.stem}_Repair.pdb"
    if not repaired_pdb.exists():
        with open(out_dir / "repair_stdout.txt", "w") as f:
            f.write(repair_stdout)
        with open(out_dir / "repair_stderr.txt", "w") as f:
            f.write(repair_stderr)
        raise RuntimeError("RepairPDB failed. Check repair_stdout.txt and repair_stderr.txt")

    # Step 2: mutation file
    mut_file = out_dir / "individual_list.txt"
    with open(mut_file, "w") as f:
        f.write(mutation + "\n")

    # Step 3: BuildModel
    build_cmd = [
        str(FOLDX_BIN),
        "--command=BuildModel",
        f"--pdb={repaired_pdb.name}",
        f"--mutant-file={mut_file.name}",
    ]
    build_res = run_cmd(build_cmd, out_dir)

    build_stdout = build_res.stdout
    build_stderr = build_res.stderr

    with open(out_dir / "build_stdout.txt", "w") as f:
        f.write(build_stdout)
    with open(out_dir / "build_stderr.txt", "w") as f:
        f.write(build_stderr)

    # Parse energy terms from stdout
    energy_df = parse_energy_from_log(build_stdout)
    if not energy_df.empty:
        energy_df.to_csv(out_dir / "energy_terms.csv", index=False)

    # Collect generated files
    generated_files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])

    summary = {
        "tool": "FoldX",
        "input_pdb": str(pdb_file),
        "sample_name": sample_name,
        "mutation": mutation,
        "results_dir": str(out_dir),
        "repaired_pdb": str(repaired_pdb) if repaired_pdb.exists() else None,
        "mutated_pdb_candidates": [f for f in generated_files if f.endswith(".pdb") and "_1" in f],
        "energy_terms_csv": str(out_dir / "energy_terms.csv") if (out_dir / "energy_terms.csv").exists() else None,
        "all_generated_files": generated_files,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if (out_dir / "energy_terms.csv").exists():
        print("\nEnergy terms:")
        print(pd.read_csv(out_dir / "energy_terms.csv").to_string(index=False))


if __name__ == "__main__":
    main()
