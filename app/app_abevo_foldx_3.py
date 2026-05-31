import json
import re
import subprocess
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from Bio.PDB import PDBParser

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RUNNERS_DIR = BASE_DIR / "runners"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def list_fastas():
    return [str(p.relative_to(BASE_DIR)) for p in sorted(DATA_DIR.rglob("*.fasta")) if "_chain_" in p.name] or [""]


def list_pdbs():
    return [str(p.relative_to(BASE_DIR)) for p in sorted(DATA_DIR.rglob("*.pdb"))] or [""]


def get_result_dir(fasta_path, pdb_path, chain):
    return RESULTS_DIR / safe_name(f"{fasta_path.stem}_{pdb_path.stem}_chain_{chain}_INTERFACE_PIPELINE")


def scan_chains(saved_pdb):
    pdb_path = BASE_DIR / saved_pdb
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_path))

    rows = []
    for model in structure:
        for chain in model:
            count = 0
            for residue in chain:
                if residue.id[0] == " " and residue.get_resname().upper() in AA3_TO_1:
                    count += 1
            rows.append({"chain": str(chain.id), "standard_residue_count": count})
        break
    return pd.DataFrame(rows)


def make_plot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.text(0.5, 0.5, "No ranked results.", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.scatter(df["abevo_score_mean"], df["foldx_total_energy"])
    for _, r in df.head(10).iterrows():
        ax.text(r["abevo_score_mean"], r["foldx_total_energy"], str(r["mapped_foldx_mutation"]), fontsize=8)

    ax.set_title("Interface-aware ABEVO vs FoldX")
    ax.set_xlabel("ABEVO score_mean")
    ax.set_ylabel("FoldX total energy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def run_interface_pipeline(saved_fasta, saved_pdb, antibody_chain, antigen_chains,
                           top_n, cutoff, restrict_mode, cdr_ranges):
    fasta_path = BASE_DIR / saved_fasta
    pdb_path = BASE_DIR / saved_pdb

    cmd = [
        "python",
        str(RUNNERS_DIR / "run_abevo_foldx_pipeline_2.py"),
        str(fasta_path),
        str(pdb_path),
        antibody_chain.strip(),
        antigen_chains.strip(),
        str(int(top_n)),
        str(float(cutoff)),
        restrict_mode,
        cdr_ranges.strip() if cdr_ranges else "",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr or res.stdout)

    out_dir = get_result_dir(fasta_path, pdb_path, antibody_chain.strip())
    summary = json.loads((out_dir / "summary.json").read_text())

    ranked = pd.read_csv(out_dir / "interface_pipeline_ranked_results.csv")
    all_df = pd.read_csv(out_dir / "interface_pipeline_all_results.csv")
    interface_df = pd.read_csv(out_dir / "interface_residues.csv")

    plot = make_plot(ranked)

    summary_text = (
        "Interface-aware ABEVO → FoldX Pipeline\n\n"
        f"FASTA: {fasta_path.name}\n"
        f"PDB: {pdb_path.name}\n"
        f"Antibody chain: {antibody_chain}\n"
        f"Antigen chains: {antigen_chains}\n"
        f"Restrict mode: {restrict_mode}\n"
        f"CDR ranges: {cdr_ranges}\n"
        f"Interface cutoff: {cutoff} Å\n"
        f"Interface residues found: {summary.get('interface_residue_count')}\n"
        f"Successful FoldX results: {summary.get('successful_foldx_results')}\n\n"
        "Meaning:\n"
        "The app now prioritizes mutations using sequence score, structural stability, "
        "and optional antigen-interface/CDR filtering."
    )

    return summary_text, ranked.head(30), plot, all_df.head(80), interface_df, json.dumps(summary, indent=2)


fastas = list_fastas()
pdbs = list_pdbs()

with gr.Blocks(title="Interface-aware ABEVO FoldX App") as demo:
    gr.Markdown("# Interface-aware ABEVO → FoldX Antibody Design App")
    gr.Markdown("Adds antigen-interface detection, optional CDR restriction, and FoldX complex-energy attempt.")

    with gr.Row():
        with gr.Column(scale=1):
            saved_fasta = gr.Dropdown(fastas, value=fastas[0], label="Choose antibody FASTA chain")
            saved_pdb = gr.Dropdown(pdbs, value=pdbs[0], label="Choose antibody-antigen complex PDB")

            antibody_chain = gr.Textbox(label="Antibody chain ID", value="H")
            antigen_chains = gr.Textbox(label="Antigen chain ID(s), comma-separated", placeholder="Example: A or C,D")

            top_n = gr.Number(label="Target number of FoldX-valid mutations", value=5, precision=0)
            cutoff = gr.Number(label="Interface cutoff distance Å", value=5.0)

            restrict_mode = gr.Dropdown(
                ["none", "interface_only", "cdr_only", "interface_and_cdr"],
                value="interface_only",
                label="Mutation restriction mode",
            )

            cdr_ranges = gr.Textbox(
                label="Optional CDR ranges using PDB numbering",
                placeholder="Example: H24-35,H50-65,H95-105",
            )

            scan_btn = gr.Button("Scan PDB Chains")
            run_btn = gr.Button("Run Interface-aware Pipeline")

        with gr.Column(scale=2):
            with gr.Tab("Summary"):
                summary = gr.Textbox(lines=14)

            with gr.Tab("Ranked Candidates"):
                ranked_table = gr.Dataframe()

            with gr.Tab("Plot"):
                plot = gr.Plot()

            with gr.Tab("All Details"):
                all_table = gr.Dataframe()

            with gr.Tab("Interface Residues"):
                interface_table = gr.Dataframe()

            with gr.Tab("PDB Chains"):
                chain_table = gr.Dataframe()

            with gr.Tab("Raw JSON"):
                raw_json = gr.Code(language="json")

    scan_btn.click(scan_chains, inputs=[saved_pdb], outputs=[chain_table])

    run_btn.click(
        run_interface_pipeline,
        inputs=[
            saved_fasta, saved_pdb, antibody_chain, antigen_chains,
            top_n, cutoff, restrict_mode, cdr_ranges
        ],
        outputs=[summary, ranked_table, plot, all_table, interface_table, raw_json],
    )

demo.launch(allowed_paths=[str(BASE_DIR), str(RESULTS_DIR)])
