import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache_interface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EE_PATH = Path("/home/ghoshlab/Desktop/Shyam/tools/efficient-evolution")
sys.path.insert(0, str(EE_PATH))
sys.path.insert(0, str(EE_PATH / "bin"))

from predict_esm import predict_esm  # noqa

FOLDX_DIR = Path("/home/ghoshlab/Desktop/Shyam/tools/foldx/foldx_Linux")
FOLDX_BIN = FOLDX_DIR / "foldx_20261231"
ROTABASE = FOLDX_DIR / "rotabase.txt"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def md5_text(x):
    return hashlib.md5(x.encode()).hexdigest()[:12]


def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line)
    return "".join(seq)


def parse_mut(mut):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", str(mut))
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def parse_ranges(text):
    """
    Input example:
    H24-35,H50-65,H95-105
    or:
    24-35,50-65,95-105
    """
    ranges = []
    if not text or not text.strip():
        return ranges

    for part in text.split(","):
        part = part.strip()
        m = re.fullmatch(r"([A-Za-z0-9]?)(\d+)-(\d+)", part)
        if not m:
            continue
        chain = m.group(1) or None
        start = int(m.group(2))
        end = int(m.group(3))
        ranges.append((chain, start, end))
    return ranges


def in_ranges(chain, pos, ranges):
    if not ranges:
        return True
    for c, s, e in ranges:
        if c is not None and c != chain:
            continue
        if s <= pos <= e:
            return True
    return False


def get_pdb_chain_map(pdb_file, antibody_chain):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_file))

    rows = []
    seen = set()

    for model in structure:
        for chain in model:
            if str(chain.id) != str(antibody_chain):
                continue

            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname().upper()
                if resname not in AA3_TO_1:
                    continue

                resseq = residue.id[1]
                icode = residue.id[2].strip()
                if icode:
                    continue

                key = (str(chain.id), resseq)
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "seq_index": len(rows),
                    "wt": AA3_TO_1[resname],
                    "pdb_chain": str(chain.id),
                    "pdb_position": int(resseq),
                    "resname3": resname,
                })
            break
        break

    return rows


def get_interface_residues(pdb_file, antibody_chain, antigen_chains, cutoff=5.0):
    """
    CPU-friendly all-atom interface detection.
    Returns antibody-chain residue positions that are within cutoff Å of antigen chain atoms.
    """
    antigen_chains = [c.strip() for c in antigen_chains.split(",") if c.strip()]
    if not antigen_chains:
        return set(), pd.DataFrame()

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_file))

    antigen_atoms = []
    antibody_residues = []

    for model in structure:
        for chain in model:
            cid = str(chain.id)

            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname().upper()
                if resname not in AA3_TO_1:
                    continue

                if cid == antibody_chain:
                    atoms = [a.coord.astype(float) for a in residue.get_atoms()]
                    antibody_residues.append({
                        "chain": cid,
                        "position": int(residue.id[1]),
                        "resname3": resname,
                        "atoms": atoms,
                    })

                elif cid in antigen_chains:
                    for atom in residue.get_atoms():
                        antigen_atoms.append(atom.coord.astype(float))
        break

    if not antigen_atoms:
        return set(), pd.DataFrame()

    antigen_arr = np.array(antigen_atoms)
    interface_positions = set()
    rows = []

    for r in antibody_residues:
        min_dist = float("inf")
        for a in r["atoms"]:
            d = np.linalg.norm(antigen_arr - a, axis=1).min()
            min_dist = min(min_dist, float(d))

        if min_dist <= cutoff:
            interface_positions.add(r["position"])
            rows.append({
                "chain": r["chain"],
                "position": r["position"],
                "residue": AA3_TO_1[r["resname3"]],
                "min_distance_to_antigen": min_dist,
            })

    return interface_positions, pd.DataFrame(rows).sort_values("position")


def run_abevo_cached(seq, fasta_stem):
    cache_file = CACHE_DIR / f"abevo_{safe_name(fasta_stem)}_{md5_text(seq)}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file), "loaded_from_cache", cache_file

    model_locations = ["esm1b_t33_650M_UR50S", "esm1v_t33_650M_UR90S_1"]
    df = predict_esm(
        seq,
        model_locations=model_locations,
        scoring_strategy="wt-marginals",
        mutation_col="mutant",
        offset_idx=0,
        nogpu=True,
        verbose=0,
    )
    df["score_mean"] = df[model_locations].mean(axis=1)
    df.to_csv(cache_file, index=False)
    return df, "computed_and_cached", cache_file


def repair_pdb_cached(pdb_file):
    cache_dir = CACHE_DIR / f"repair_{safe_name(pdb_file.stem)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repaired = cache_dir / f"{pdb_file.stem}_Repair.pdb"

    if repaired.exists():
        return repaired, "loaded_from_cache"

    shutil.copy2(pdb_file, cache_dir / pdb_file.name)
    shutil.copy2(ROTABASE, cache_dir / "rotabase.txt")

    cmd = [str(FOLDX_BIN), "--command=RepairPDB", f"--pdb={pdb_file.name}"]
    res = subprocess.run(cmd, cwd=str(cache_dir), capture_output=True, text=True)

    (cache_dir / "repair_stdout.txt").write_text(res.stdout)
    (cache_dir / "repair_stderr.txt").write_text(res.stderr)

    if not repaired.exists():
        raise RuntimeError("FoldX RepairPDB failed:\n" + res.stdout + "\n" + res.stderr)

    return repaired, "computed_and_cached"


def parse_total_from_stdout(stdout):
    for line in stdout.splitlines():
        if line.strip().startswith("Total") and "=" in line:
            try:
                return float(line.split("=")[1].strip())
            except Exception:
                pass
    return None


def run_foldx_buildmodel(pdb_file, repaired_pdb, foldx_mut, out_dir):
    cache_file = CACHE_DIR / f"foldx_{safe_name(pdb_file.stem)}_{safe_name(foldx_mut)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    run_dir = out_dir / f"foldx_{safe_name(foldx_mut)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(repaired_pdb, run_dir / repaired_pdb.name)
    shutil.copy2(ROTABASE, run_dir / "rotabase.txt")

    (run_dir / "individual_list.txt").write_text(foldx_mut + ";\n")

    cmd = [
        str(FOLDX_BIN),
        "--command=BuildModel",
        f"--pdb={repaired_pdb.name}",
        "--mutant-file=individual_list.txt",
    ]

    res = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)

    (run_dir / "build_stdout.txt").write_text(res.stdout)
    (run_dir / "build_stderr.txt").write_text(res.stderr)

    total = parse_total_from_stdout(res.stdout)

    mutated_candidates = sorted(run_dir.glob("*_1.pdb")) + sorted(run_dir.glob("*Repair_1.pdb"))
    mutated_pdb = str(mutated_candidates[0]) if mutated_candidates else None

    result = {
        "foldx_mutation": foldx_mut,
        "foldx_total_energy": total,
        "mutated_pdb": mutated_pdb,
        "build_status": "OK" if total is not None else "FAILED_OR_UNPARSED",
        "run_dir": str(run_dir),
    }

    cache_file.write_text(json.dumps(result, indent=2))
    return result


def run_foldx_analyse_complex(pdb_path, antibody_chain, antigen_chains, out_dir, label):
    """
    Attempts FoldX AnalyseComplex. If parsing fails, output remains None.
    """
    if not antigen_chains:
        return None, "no_antigen_chains"

    antigen_clean = ",".join([c.strip() for c in antigen_chains.split(",") if c.strip()])
    if not antigen_clean:
        return None, "no_antigen_chains"

    run_dir = out_dir / f"analyse_complex_{safe_name(label)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = Path(pdb_path)
    shutil.copy2(pdb_path, run_dir / pdb_path.name)
    shutil.copy2(ROTABASE, run_dir / "rotabase.txt")

    chains_arg = f"{antibody_chain},{antigen_clean}"

    cmd = [
        str(FOLDX_BIN),
        "--command=AnalyseComplex",
        f"--pdb={pdb_path.name}",
        f"--analyseComplexChains={chains_arg}",
    ]

    res = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)

    (run_dir / "analyse_stdout.txt").write_text(res.stdout)
    (run_dir / "analyse_stderr.txt").write_text(res.stderr)

    # Simple parser: search for phrases and numeric values in fxout/stdout.
    text = res.stdout + "\n" + res.stderr
    for fx in run_dir.glob("*.fxout"):
        try:
            text += "\n" + fx.read_text(errors="ignore")
        except Exception:
            pass

    binding_energy = None
    for line in text.splitlines():
        low = line.lower()
        if "interaction" in low or "complex" in low or "energy" in low:
            nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", line)
            if nums:
                try:
                    binding_energy = float(nums[-1])
                except Exception:
                    pass

    status = "OK" if binding_energy is not None else "UNPARSED_OR_FAILED"
    return binding_energy, status


def main():
    if len(sys.argv) < 9:
        print(
            "Usage: python run_interface_abevo_foldx.py "
            "<fasta> <pdb> <antibody_chain> <antigen_chains> <top_n> "
            "<interface_cutoff> <restrict_mode> <cdr_ranges>"
        )
        sys.exit(1)

    fasta = Path(sys.argv[1]).resolve()
    pdb = Path(sys.argv[2]).resolve()
    antibody_chain = sys.argv[3].strip()
    antigen_chains = sys.argv[4].strip()
    top_n = int(sys.argv[5])
    interface_cutoff = float(sys.argv[6])
    restrict_mode = sys.argv[7].strip()
    cdr_ranges_text = sys.argv[8].strip()

    seq = read_fasta(fasta)
    if len(seq) > 1024:
        raise ValueError("Selected FASTA is too long. Use a single chain FASTA.")

    out_dir = RESULTS_DIR / safe_name(
        f"{fasta.stem}_{pdb.stem}_chain_{antibody_chain}_INTERFACE_PIPELINE"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_map = get_pdb_chain_map(pdb, antibody_chain)
    if not pdb_map:
        raise ValueError(f"No residues found for antibody chain {antibody_chain}")

    interface_positions, interface_df = get_interface_residues(
        pdb,
        antibody_chain,
        antigen_chains,
        cutoff=interface_cutoff,
    )
    interface_df.to_csv(out_dir / "interface_residues.csv", index=False)

    cdr_ranges = parse_ranges(cdr_ranges_text)

    abevo_df, abevo_cache_status, abevo_cache_file = run_abevo_cached(seq, fasta.stem)
    abevo_df.to_csv(out_dir / "abevo_all_mutations.csv", index=False)

    repaired_pdb, repair_cache_status = repair_pdb_cached(pdb)

    candidate_pool = max(top_n * 100, 300)
    top_df = abevo_df.sort_values("score_mean", ascending=False).head(candidate_pool)

    rows = []
    foldx_ready = []

    for _, row in top_df.iterrows():
        abevo_mut = str(row["mutant"])
        parsed = parse_mut(abevo_mut)
        if parsed is None:
            continue

        wt, seq_index, mt = parsed
        if seq_index >= len(pdb_map):
            continue

        mapped = pdb_map[seq_index]
        pdb_pos = int(mapped["pdb_position"])

        if wt != mapped["wt"]:
            rows.append({
                "abevo_mutation": abevo_mut,
                "mapped_foldx_mutation": None,
                "sequence_index_0based": seq_index,
                "sequence_position_1based": seq_index + 1,
                "pdb_chain": antibody_chain,
                "pdb_position": pdb_pos,
                "wt": wt,
                "mutant": mt,
                "abevo_score_mean": row["score_mean"],
                "mapping_status": f"WT mismatch: ABEVO {wt}, PDB {mapped['wt']}",
            })
            continue

        is_interface = pdb_pos in interface_positions if antigen_chains else False
        is_cdr = in_ranges(antibody_chain, pdb_pos, cdr_ranges)

        keep = True
        if restrict_mode == "interface_only":
            keep = is_interface
        elif restrict_mode == "cdr_only":
            keep = is_cdr
        elif restrict_mode == "interface_and_cdr":
            keep = is_interface and is_cdr

        if not keep:
            rows.append({
                "abevo_mutation": abevo_mut,
                "mapped_foldx_mutation": None,
                "sequence_index_0based": seq_index,
                "sequence_position_1based": seq_index + 1,
                "pdb_chain": antibody_chain,
                "pdb_position": pdb_pos,
                "wt": wt,
                "mutant": mt,
                "abevo_score_mean": row["score_mean"],
                "is_interface": is_interface,
                "is_cdr": is_cdr,
                "mapping_status": f"filtered_by_{restrict_mode}",
            })
            continue

        foldx_mut = f"{wt}{antibody_chain}{pdb_pos}{mt}"

        foldx_ready.append({
            "abevo_mutation": abevo_mut,
            "mapped_foldx_mutation": foldx_mut,
            "sequence_index_0based": seq_index,
            "sequence_position_1based": seq_index + 1,
            "pdb_chain": antibody_chain,
            "pdb_position": pdb_pos,
            "wt": wt,
            "mutant": mt,
            "abevo_score_mean": row["score_mean"],
            "is_interface": is_interface,
            "is_cdr": is_cdr,
        })

        if len(foldx_ready) >= top_n:
            break

    for item in foldx_ready:
        build_res = run_foldx_buildmodel(
            pdb_file=pdb,
            repaired_pdb=repaired_pdb,
            foldx_mut=item["mapped_foldx_mutation"],
            out_dir=out_dir,
        )

        wt_binding, wt_binding_status = run_foldx_analyse_complex(
            pdb_path=repaired_pdb,
            antibody_chain=antibody_chain,
            antigen_chains=antigen_chains,
            out_dir=out_dir,
            label="WT",
        )

        mut_binding = None
        mut_binding_status = "no_mutated_pdb"
        if build_res.get("mutated_pdb"):
            mut_binding, mut_binding_status = run_foldx_analyse_complex(
                pdb_path=build_res["mutated_pdb"],
                antibody_chain=antibody_chain,
                antigen_chains=antigen_chains,
                out_dir=out_dir,
                label=item["mapped_foldx_mutation"],
            )

        delta_binding = None
        if wt_binding is not None and mut_binding is not None:
            delta_binding = mut_binding - wt_binding

        rows.append({
            **item,
            "foldx_total_energy": build_res.get("foldx_total_energy"),
            "mutated_pdb": build_res.get("mutated_pdb"),
            "wt_binding_energy": wt_binding,
            "mutant_binding_energy": mut_binding,
            "delta_binding_energy": delta_binding,
            "binding_status": mut_binding_status,
            "mapping_status": build_res.get("build_status"),
        })

    all_df = pd.DataFrame(rows)

    expected = [
        "abevo_mutation", "mapped_foldx_mutation", "sequence_index_0based",
        "sequence_position_1based", "pdb_chain", "pdb_position", "wt", "mutant",
        "abevo_score_mean", "is_interface", "is_cdr", "foldx_total_energy",
        "wt_binding_energy", "mutant_binding_energy", "delta_binding_energy",
        "binding_status", "mapping_status", "mutated_pdb"
    ]
    for col in expected:
        if col not in all_df.columns:
            all_df[col] = None

    valid = all_df.dropna(subset=["foldx_total_energy"]).copy()
    if not valid.empty:
        valid["abevo_rank"] = valid["abevo_score_mean"].rank(ascending=False)
        valid["foldx_rank"] = valid["foldx_total_energy"].rank(ascending=True)

        if valid["delta_binding_energy"].notna().any():
            valid["binding_rank"] = valid["delta_binding_energy"].rank(ascending=True)
        else:
            valid["binding_rank"] = 0

        valid["combined_rank_score"] = (
            valid["abevo_rank"] + valid["foldx_rank"] + valid["binding_rank"]
        )
        valid = valid.sort_values("combined_rank_score")

    all_df.to_csv(out_dir / "interface_pipeline_all_results.csv", index=False)
    valid.to_csv(out_dir / "interface_pipeline_ranked_results.csv", index=False)

    summary = {
        "tool": "interface_aware_ABEVO_FoldX_pipeline",
        "input_fasta": str(fasta),
        "input_pdb": str(pdb),
        "antibody_chain": antibody_chain,
        "antigen_chains": antigen_chains,
        "top_n_requested": top_n,
        "interface_cutoff": interface_cutoff,
        "restrict_mode": restrict_mode,
        "cdr_ranges": cdr_ranges_text,
        "sequence_length": len(seq),
        "pdb_chain_residue_count": len(pdb_map),
        "interface_residue_count": len(interface_positions),
        "successful_foldx_results": int(len(valid)),
        "abevo_cache_status": abevo_cache_status,
        "repair_cache_status": repair_cache_status,
        "results_dir": str(out_dir),
        "ranked_csv": str(out_dir / "interface_pipeline_ranked_results.csv"),
        "all_csv": str(out_dir / "interface_pipeline_all_results.csv"),
        "interface_residues_csv": str(out_dir / "interface_residues.csv"),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
