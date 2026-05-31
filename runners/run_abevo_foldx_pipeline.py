import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"
ABEVO_CACHE_DIR = CACHE_DIR / "abevo"
FOLDX_CACHE_DIR = CACHE_DIR / "foldx"
REPAIR_CACHE_DIR = CACHE_DIR / "repaired_pdbs"

for d in [RESULTS_DIR, CACHE_DIR, ABEVO_CACHE_DIR, FOLDX_CACHE_DIR, REPAIR_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EE_PATH = Path("/home/ghoshlab/Desktop/Shyam/tools/efficient-evolution")
sys.path.insert(0, str(EE_PATH))
sys.path.insert(0, str(EE_PATH / "bin"))

from predict_esm import predict_esm  # noqa: E402

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


def file_hash(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def text_hash(text: str):
    return hashlib.md5(text.encode()).hexdigest()[:12]


def read_fasta(path: Path):
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


def get_pdb_chain_map(pdb_file: Path, chain_id: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_file))
    rows, seen = [], set()

    for model in structure:
        for chain in model:
            if str(chain.id) != str(chain_id):
                continue

            for residue in chain:
                if residue.id[0] != " ":
                    continue

                resname = residue.get_resname().upper()
                if resname not in AA3_TO_1:
                    continue

                resseq = residue.id[1]
                icode = residue.id[2].strip()

                # Skip insertion-coded residues for first stable version
                if icode:
                    continue

                key = (chain.id, resseq)
                if key in seen:
                    continue

                seen.add(key)

                rows.append({
                    "seq_index": len(rows),
                    "wt": AA3_TO_1[resname],
                    "chain": str(chain.id),
                    "pdb_position": str(resseq),
                })

            break
        break

    return rows


def load_or_run_abevo(seq: str, fasta: Path, out_dir: Path):
    cache_key = f"{safe_name(fasta.stem)}_{text_hash(seq)}"
    cache_csv = ABEVO_CACHE_DIR / f"{cache_key}_abevo_all_mutations.csv"

    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        cache_status = "loaded_from_cache"
    else:
        model_locations = [
            "esm1b_t33_650M_UR50S",
            "esm1v_t33_650M_UR90S_1",
        ]

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
        df.to_csv(cache_csv, index=False)
        cache_status = "computed_and_cached"

    shutil.copy2(cache_csv, out_dir / "abevo_all_mutations.csv")

    return df, cache_status, str(cache_csv)


def repair_pdb_once(pdb_file: Path):
    pdb_key = f"{safe_name(pdb_file.stem)}_{file_hash(pdb_file)}"
    repair_dir = REPAIR_CACHE_DIR / pdb_key
    repair_dir.mkdir(parents=True, exist_ok=True)

    repaired_pdb = repair_dir / f"{pdb_file.stem}_Repair.pdb"

    if repaired_pdb.exists():
        return repaired_pdb, "loaded_from_cache", repair_dir

    shutil.copy2(pdb_file, repair_dir / pdb_file.name)
    shutil.copy2(ROTABASE, repair_dir / "rotabase.txt")

    repair = subprocess.run(
        [str(FOLDX_BIN), "--command=RepairPDB", f"--pdb={pdb_file.name}"],
        cwd=str(repair_dir),
        capture_output=True,
        text=True,
    )

    (repair_dir / "repair_stdout.txt").write_text(repair.stdout)
    (repair_dir / "repair_stderr.txt").write_text(repair.stderr)

    if not repaired_pdb.exists():
        raise RuntimeError(
            f"RepairPDB failed for {pdb_file}\n\nSTDOUT:\n{repair.stdout}\n\nSTDERR:\n{repair.stderr}"
        )

    return repaired_pdb, "computed_and_cached", repair_dir


def load_foldx_cache(pdb_file: Path):
    pdb_key = f"{safe_name(pdb_file.stem)}_{file_hash(pdb_file)}"
    cache_csv = FOLDX_CACHE_DIR / f"{pdb_key}_foldx_mutation_cache.csv"

    if cache_csv.exists():
        return pd.read_csv(cache_csv), cache_csv

    df = pd.DataFrame(columns=[
        "mapped_foldx_mutation",
        "foldx_total_energy",
        "foldx_status",
        "foldx_cache_status",
    ])
    return df, cache_csv


def save_foldx_cache(cache_df: pd.DataFrame, cache_csv: Path):
    cache_df = cache_df.drop_duplicates(subset=["mapped_foldx_mutation"], keep="last")
    cache_df.to_csv(cache_csv, index=False)


def parse_foldx_energies_from_fxout(run_dir: Path, mutations):
    """
    Tries to parse FoldX batch output.
    FoldX often creates files such as Dif_*.fxout or Average_*.fxout.
    If parsing fails, returns empty dict.
    """
    energies = {}

    fxouts = list(run_dir.glob("*.fxout"))
    if not fxouts:
        return energies

    preferred = []
    for p in fxouts:
        name = p.name.lower()
        if name.startswith("dif_") or name.startswith("average_") or "raw" in name:
            preferred.append(p)

    candidates = preferred if preferred else fxouts

    for fx in candidates:
        try:
            lines = fx.read_text(errors="ignore").splitlines()
        except Exception:
            continue

        data_lines = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "Pdb" in s and "total" in s.lower():
                continue
            if re.search(r"[-+]?\d+\.\d+", s):
                data_lines.append(s)

        # Simple robust strategy:
        # take last numeric value from each likely data line
        numeric_values = []
        for line in data_lines:
            nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", line)
            if nums:
                try:
                    numeric_values.append(float(nums[-1]))
                except Exception:
                    pass

        if len(numeric_values) >= len(mutations):
            for mut, val in zip(mutations, numeric_values[:len(mutations)]):
                energies[mut] = val
            return energies

    return energies


def run_foldx_batch(pdb_file: Path, repaired_pdb: Path, mutations, out_dir: Path):
    """
    Runs FoldX BuildModel once for a batch of mutations.
    Also caches per-mutation results permanently.
    """
    cache_df, cache_csv = load_foldx_cache(pdb_file)

    cached = {}
    if not cache_df.empty and "mapped_foldx_mutation" in cache_df.columns:
        for _, r in cache_df.iterrows():
            cached[str(r["mapped_foldx_mutation"])] = {
                "energy": r.get("foldx_total_energy"),
                "status": r.get("foldx_status", "OK"),
                "cache_status": "loaded_from_cache",
            }

    missing = [m for m in mutations if m not in cached]

    batch_results = {}

    for mut in mutations:
        if mut in cached:
            batch_results[mut] = cached[mut]

    if not missing:
        return batch_results, "all_loaded_from_cache"

    batch_key = text_hash("|".join(missing))
    batch_dir = out_dir / f"foldx_batch_{batch_key}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(repaired_pdb, batch_dir / repaired_pdb.name)
    shutil.copy2(ROTABASE, batch_dir / "rotabase.txt")

    mut_file = batch_dir / "individual_list.txt"
    mut_file.write_text("\n".join([m + ";" for m in missing]) + "\n")

    build = subprocess.run(
        [
            str(FOLDX_BIN),
            "--command=BuildModel",
            f"--pdb={repaired_pdb.name}",
            "--mutant-file=individual_list.txt",
        ],
        cwd=str(batch_dir),
        capture_output=True,
        text=True,
    )

    (batch_dir / "build_stdout.txt").write_text(build.stdout)
    (batch_dir / "build_stderr.txt").write_text(build.stderr)

    parsed_energies = parse_foldx_energies_from_fxout(batch_dir, missing)

    new_cache_rows = []

    for mut in missing:
        energy = parsed_energies.get(mut)

        # fallback: if only one mutation, try stdout Total
        if energy is None and len(missing) == 1:
            for line in build.stdout.splitlines():
                if line.strip().startswith("Total") and "=" in line:
                    try:
                        energy = float(line.split("=")[1].strip())
                    except Exception:
                        pass

        status = "OK" if energy is not None else "FoldX batch failed_or_unparsed"

        batch_results[mut] = {
            "energy": energy,
            "status": status,
            "cache_status": "computed_and_cached" if energy is not None else "computed_but_failed",
        }

        if energy is not None:
            new_cache_rows.append({
                "mapped_foldx_mutation": mut,
                "foldx_total_energy": energy,
                "foldx_status": status,
                "foldx_cache_status": "computed_and_cached",
            })

    if new_cache_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_cache_rows)], ignore_index=True)
        save_foldx_cache(cache_df, cache_csv)

    return batch_results, "partial_cache_or_batch_run"


def main():
    if len(sys.argv) < 5:
        print("Usage: python run_abevo_foldx_pipeline.py <fasta> <pdb> <chain_id> <top_n>")
        sys.exit(1)

    fasta = Path(sys.argv[1]).resolve()
    pdb = Path(sys.argv[2]).resolve()
    chain_id = sys.argv[3].strip()
    top_n = int(sys.argv[4])

    if not fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta}")
    if not pdb.exists():
        raise FileNotFoundError(f"PDB not found: {pdb}")
    if not FOLDX_BIN.exists():
        raise FileNotFoundError(f"FoldX binary not found: {FOLDX_BIN}")
    if not ROTABASE.exists():
        raise FileNotFoundError(f"rotabase.txt not found: {ROTABASE}")

    seq = read_fasta(fasta)

    if len(seq) > 1024:
        raise ValueError(
            f"Selected FASTA is too long: {len(seq)} residues. "
            "Please choose a single-chain FASTA such as data/1IGT_fasta/1IGT_chain_B.fasta."
        )

    pdb_map = get_pdb_chain_map(pdb, chain_id)

    if not pdb_map:
        raise ValueError(f"No valid residues found for chain {chain_id}")

    out_dir = RESULTS_DIR / safe_name(f"{fasta.stem}_{pdb.stem}_chain_{chain_id}_PIPELINE_V2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ABEVO cache
    df, abevo_cache_status, abevo_cache_csv = load_or_run_abevo(seq, fasta, out_dir)

    # 2. RepairPDB cache
    repaired_pdb, repair_cache_status, repair_dir = repair_pdb_once(pdb)

    # Candidate pool
    candidate_pool = max(top_n * 100, 200)
    top_df = df.sort_values("score_mean", ascending=False).head(candidate_pool)

    rows = []
    foldx_mutations = []
    foldx_to_abevo = {}

    for _, row in top_df.iterrows():
        abevo_mut = str(row["mutant"])
        parsed = parse_mut(abevo_mut)

        if parsed is None:
            rows.append({
                "abevo_mutation": abevo_mut,
                "mapped_foldx_mutation": None,
                "sequence_index_0based": None,
                "sequence_position_1based": None,
                "pdb_chain": chain_id,
                "pdb_position": None,
                "wt": None,
                "mutant": None,
                "abevo_score_mean": row.get("score_mean"),
                "foldx_total_energy": None,
                "mapping_status": "could not parse ABEVO mutation",
            })
            continue

        wt, seq_index, mt = parsed

        if seq_index >= len(pdb_map):
            rows.append({
                "abevo_mutation": abevo_mut,
                "mapped_foldx_mutation": None,
                "sequence_index_0based": seq_index,
                "sequence_position_1based": seq_index + 1,
                "pdb_chain": chain_id,
                "pdb_position": None,
                "wt": wt,
                "mutant": mt,
                "abevo_score_mean": row.get("score_mean"),
                "foldx_total_energy": None,
                "mapping_status": "sequence index outside PDB chain",
            })
            continue

        pdb_res = pdb_map[seq_index]

        if wt != pdb_res["wt"]:
            rows.append({
                "abevo_mutation": abevo_mut,
                "mapped_foldx_mutation": None,
                "sequence_index_0based": seq_index,
                "sequence_position_1based": seq_index + 1,
                "pdb_chain": chain_id,
                "pdb_position": pdb_res["pdb_position"],
                "wt": wt,
                "mutant": mt,
                "abevo_score_mean": row.get("score_mean"),
                "foldx_total_energy": None,
                "mapping_status": f"WT mismatch: ABEVO {wt}, PDB {pdb_res['wt']}",
            })
            continue

        foldx_mut = f"{wt}{chain_id}{pdb_res['pdb_position']}{mt}"

        foldx_mutations.append(foldx_mut)
        foldx_to_abevo[foldx_mut] = {
            "abevo_mutation": abevo_mut,
            "mapped_foldx_mutation": foldx_mut,
            "sequence_index_0based": seq_index,
            "sequence_position_1based": seq_index + 1,
            "pdb_chain": chain_id,
            "pdb_position": pdb_res["pdb_position"],
            "wt": wt,
            "mutant": mt,
            "abevo_score_mean": row.get("score_mean"),
        }

        if len(foldx_mutations) >= top_n:
            break

    # 3. FoldX batch mutation file + persistent FoldX cache
    batch_status = "no_foldx_candidates"
    if foldx_mutations:
        foldx_batch_results, batch_status = run_foldx_batch(
            pdb_file=pdb,
            repaired_pdb=repaired_pdb,
            mutations=foldx_mutations,
            out_dir=out_dir,
        )

        for mut in foldx_mutations:
            base = foldx_to_abevo[mut]
            res = foldx_batch_results.get(mut, {})
            rows.append({
                **base,
                "foldx_total_energy": res.get("energy"),
                "mapping_status": "OK" if res.get("energy") is not None else res.get("status"),
                "foldx_cache_status": res.get("cache_status"),
            })

    all_results = pd.DataFrame(rows)

    expected_cols = [
        "abevo_mutation",
        "mapped_foldx_mutation",
        "sequence_index_0based",
        "sequence_position_1based",
        "pdb_chain",
        "pdb_position",
        "wt",
        "mutant",
        "abevo_score_mean",
        "foldx_total_energy",
        "mapping_status",
        "foldx_cache_status",
    ]

    for col in expected_cols:
        if col not in all_results.columns:
            all_results[col] = None

    valid = all_results.dropna(subset=["foldx_total_energy"]).copy()

    if not valid.empty:
        valid["abevo_rank"] = valid["abevo_score_mean"].rank(ascending=False)
        valid["foldx_rank"] = valid["foldx_total_energy"].rank(ascending=True)
        valid["combined_rank_score"] = valid["abevo_rank"] + valid["foldx_rank"]
        valid = valid.sort_values("combined_rank_score")

    all_results.to_csv(out_dir / "pipeline_all_results.csv", index=False)
    valid.to_csv(out_dir / "pipeline_ranked_results.csv", index=False)

    summary = {
        "tool": "ABEVO_to_FoldX_pipeline_v2_cached_batch",
        "input_fasta": str(fasta),
        "input_pdb": str(pdb),
        "chain_id": chain_id,
        "sequence_length": len(seq),
        "pdb_chain_residue_count": len(pdb_map),
        "top_n_requested": top_n,
        "candidate_pool_checked": int(candidate_pool),
        "successful_foldx_results": int(len(valid)),
        "abevo_cache_status": abevo_cache_status,
        "abevo_cache_csv": abevo_cache_csv,
        "repair_cache_status": repair_cache_status,
        "repair_cache_dir": str(repair_dir),
        "foldx_batch_status": batch_status,
        "results_dir": str(out_dir),
        "abevo_all_mutations_csv": str(out_dir / "abevo_all_mutations.csv"),
        "pipeline_all_results_csv": str(out_dir / "pipeline_all_results.csv"),
        "pipeline_ranked_results_csv": str(out_dir / "pipeline_ranked_results.csv"),
        "persistent_cache_dir": str(CACHE_DIR),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
