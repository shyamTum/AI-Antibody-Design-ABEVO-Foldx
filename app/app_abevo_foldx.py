import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from Bio.PDB import PDBParser

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RUNNERS_DIR = BASE_DIR / "runners"
REPORT_DIR = RESULTS_DIR / "report_exports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def list_fastas():
    return [str(p.relative_to(BASE_DIR)) for p in sorted(DATA_DIR.rglob("*.fasta"))] or [""]


def list_pdbs():
    return [str(p.relative_to(BASE_DIR)) for p in sorted(DATA_DIR.rglob("*.pdb"))] or [""]


def scan_pdb_sites(saved_pdb):
    pdb_path = BASE_DIR / saved_pdb
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_path))

    rows, seen = [], set()
    for model in structure:
        for chain in model:
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
                key = (chain.id, resseq)
                if key in seen:
                    continue
                seen.add(key)

                wt = AA3_TO_1[resname]
                example = f"{wt}{chain.id}{resseq}V" if wt != "V" else f"{wt}{chain.id}{resseq}A"

                rows.append({
                    "wt": wt,
                    "chain": chain.id,
                    "position": resseq,
                    "example_foldx_mutation": example,
                })

    df = pd.DataFrame(rows).sort_values(["chain", "position"])
    return f"Found {len(df)} valid FoldX-compatible residue sites.", df.head(100)


def make_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["abevo_score_mean"], df["foldx_total_energy"])

    for _, row in df.head(10).iterrows():
        ax.text(
            row["abevo_score_mean"],
            row["foldx_total_energy"],
            str(row["mapped_foldx_mutation"]),
            fontsize=8,
        )

    ax.set_title("ABEVO score vs FoldX energy")
    ax.set_xlabel("ABEVO score_mean (higher preferred)")
    ax.set_ylabel("FoldX total energy (lower preferred)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_bar(df):
    plot_df = df.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["mapped_foldx_mutation"].astype(str), plot_df["combined_rank_score"])
    ax.set_title("Combined ABEVO + FoldX Ranking")
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Combined rank score (lower is better)")
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    return fig


def save_pdf(summary_text, main_df, extra_df, fig):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_DIR / f"ABEVO_FoldX_pipeline_report_{ts}.pdf"

    with PdfPages(pdf_path) as pdf:
        fig1, ax1 = plt.subplots(figsize=(11, 8.5))
        ax1.axis("off")
        ax1.text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=11, family="monospace")
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight")

        for title, df in [("Main Table", main_df), ("Extra Details", extra_df)]:
            if df is not None and not df.empty:
                fig2, ax2 = plt.subplots(figsize=(11, 8.5))
                ax2.axis("off")
                table = ax2.table(cellText=df.head(20).values, colLabels=list(df.columns), loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 1.25)
                ax2.set_title(title)
                pdf.savefig(fig2, bbox_inches="tight")
                plt.close(fig2)

    return str(pdf_path)


def run_pipeline(saved_fasta, saved_pdb, chain_id, top_n):
    fasta_path = BASE_DIR / saved_fasta
    pdb_path = BASE_DIR / saved_pdb
    chain_id = chain_id.strip()
    top_n = int(top_n)

    if not chain_id:
        raise ValueError("Please enter PDB chain ID, e.g., A or B.")

    cmd = [
        "python",
        str(RUNNERS_DIR / "run_abevo_foldx_pipeline.py"),
        str(fasta_path),
        str(pdb_path),
        chain_id,
        str(top_n),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    result_dir = RESULTS_DIR / safe_name(f"{fasta_path.stem}_{pdb_path.stem}_chain_{chain_id}_PIPELINE_V2")
    summary = json.loads((result_dir / "summary.json").read_text())

    ranked_csv = result_dir / "pipeline_ranked_results.csv"
    all_csv = result_dir / "pipeline_all_results.csv"

    ranked_df = pd.read_csv(ranked_csv) if ranked_csv.exists() else pd.DataFrame()
    all_df = pd.read_csv(all_csv) if all_csv.exists() else pd.DataFrame()

    if ranked_df.empty:
        main_df = all_df.head(20)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid FoldX-ranked mutations. Check mapping details.", ha="center")
    else:
        main_df = ranked_df.head(10)
        fig = make_scatter(ranked_df)

    summary_text = (
        "ABEVO → FoldX Pipeline Report\n\n"
        f"FASTA: {fasta_path.name}\n"
        f"PDB: {pdb_path.name}\n"
        f"PDB chain ID: {chain_id}\n"
        f"Top N requested: {top_n}\n"
        f"Successful FoldX results: {summary.get('successful_foldx_results')}\n\n"
        "Interpretation:\n"
        "1. Antibody-Evolution first scores all single-point mutations from the sequence.\n"
        "2. Top ABEVO mutations are mapped to the selected PDB chain.\n"
        "3. FoldX evaluates the mapped mutations structurally.\n"
        "4. Final candidates are ranked using both sequence/evolution score and FoldX energy."
    )

    pdf = save_pdf(summary_text, main_df, all_df.head(25), fig)

    return summary_text, main_df, fig, all_df.head(25), json.dumps(summary, indent=2), pdf


fastas = list_fastas()
pdbs = list_pdbs()

with gr.Blocks(title="ABEVO → FoldX Mentor App") as demo:
    gr.Markdown("# ABEVO → FoldX Antibody Mutation Prioritization App")
    gr.Markdown("This focused app runs everything from the frontend: ABEVO mutation scoring → FoldX structural filtering → ranked report.")

    with gr.Row():
        with gr.Column(scale=1):
            saved_fasta = gr.Dropdown(fastas, value=fastas[0], label="Choose FASTA chain")
            saved_pdb = gr.Dropdown(pdbs, value=pdbs[0], label="Choose matching PDB")
            chain_id = gr.Textbox(label="PDB Chain ID", placeholder="Example: A or B", value="B")
            top_n = gr.Number(label="Top N ABEVO mutations to send to FoldX", value=5, precision=0)

            scan_btn = gr.Button("Scan Valid FoldX Sites")
            run_btn = gr.Button("Run Full ABEVO → FoldX Pipeline")

            report_file = gr.File(label="Download PDF Report")

        with gr.Column(scale=2):
            with gr.Tab("Important Summary"):
                summary = gr.Textbox(label="Key Findings", lines=14)

            with gr.Tab("Ranked Candidates"):
                main_table = gr.Dataframe(label="Final Ranked Mutations")

            with gr.Tab("Main Plot"):
                plot = gr.Plot(label="ABEVO vs FoldX Plot")

            with gr.Tab("Mapping Details"):
                extra = gr.Dataframe(label="All Mapping/FoldX Details")

            with gr.Tab("Valid FoldX Sites"):
                scan_summary = gr.Textbox(label="Scan Summary", lines=3)
                scan_table = gr.Dataframe(label="Available PDB Residue Sites")

            with gr.Tab("Raw JSON"):
                raw_json = gr.Code(label="Summary JSON", language="json")

    scan_btn.click(scan_pdb_sites, inputs=[saved_pdb], outputs=[scan_summary, scan_table])

    run_btn.click(
        run_pipeline,
        inputs=[saved_fasta, saved_pdb, chain_id, top_n],
        outputs=[summary, main_table, plot, extra, raw_json, report_file],
    )

demo.launch(
    allowed_paths=[
        str(BASE_DIR),
        str(RESULTS_DIR),
        str(REPORT_DIR),
    ]
)
