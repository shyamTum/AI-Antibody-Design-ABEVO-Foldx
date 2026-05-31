import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import umap
from Bio.PDB import PDBParser

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
RUNNERS_DIR = BASE_DIR / "runners"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TMP_DIR = BASE_DIR / "tmp"
REPORT_DIR = RESULTS_DIR / "report_exports"

TMP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# path helpers
# -------------------------
def runner_path(*names):
    for name in names:
        p = RUNNERS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these runners found: {names}")


def list_saved_fastas():
    fastas = sorted(DATA_DIR.rglob("*.fasta"))
    rels = [str(f.relative_to(BASE_DIR)) for f in fastas]
    return rels if rels else [""]


def list_saved_pdbs():
    pdbs = sorted(DATA_DIR.rglob("*.pdb"))
    rels = [str(f.relative_to(BASE_DIR)) for f in pdbs]
    return rels if rels else [""]


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)


def write_fasta(seq: str, out_path: Path):
    seq = seq.strip().replace(" ", "").replace("\n", "")
    with open(out_path, "w") as f:
        f.write(">input_sequence\n")
        f.write(seq + "\n")


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def run_subprocess(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)


# -------------------------
# input helpers
# -------------------------
def choose_sequence(input_mode, pasted_sequence, saved_fasta_rel):
    if input_mode == "Paste sequence":
        seq = (pasted_sequence or "").strip().replace(" ", "").replace("\n", "")
        if not seq:
            raise ValueError("Please paste a sequence.")
        fasta_path = TMP_DIR / "input_sequence.fasta"
        write_fasta(seq, fasta_path)
        return fasta_path, "input_sequence", seq
    else:
        if not saved_fasta_rel:
            raise ValueError("Please choose a saved FASTA.")
        fasta_path = BASE_DIR / saved_fasta_rel
        if not fasta_path.exists():
            raise FileNotFoundError(f"Saved FASTA not found: {fasta_path}")
        return fasta_path, Path(saved_fasta_rel).stem, read_fasta(fasta_path)


def choose_pdb(pdb_mode, uploaded_pdb, saved_pdb_rel):
    if pdb_mode == "Upload PDB":
        if not uploaded_pdb:
            raise ValueError("Please upload a PDB.")
        return Path(uploaded_pdb)
    else:
        if not saved_pdb_rel:
            raise ValueError("Please choose a saved PDB.")
        pdb_path = BASE_DIR / saved_pdb_rel
        if not pdb_path.exists():
            raise FileNotFoundError(f"Saved PDB not found: {pdb_path}")
        return pdb_path


# -------------------------
# mutation helpers
# -------------------------
def parse_seq_mutation(mut: str):
    mut = mut.strip().replace(";", "")
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut)
    if not m:
        raise ValueError("Sequence mutation must look like A23V")
    wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
    return wt, pos, mt


def apply_seq_mutation(seq: str, mutation: str):
    wt, pos, mt = parse_seq_mutation(mutation)
    idx = pos - 1
    if idx < 0 or idx >= len(seq):
        raise ValueError(f"Mutation position {pos} is out of range for sequence length {len(seq)}")
    if seq[idx] != wt:
        raise ValueError(f"Mutation WT mismatch: sequence has {seq[idx]} at position {pos}, not {wt}")
    return seq[:idx] + mt + seq[idx + 1:]


# -------------------------
# FoldX residue scanning
# -------------------------
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}


def scan_pdb_mutations(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_path))

    rows = []
    seen = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                # standard residues only
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname().upper()
                if resname not in AA3_TO_1:
                    continue

                chain_id = str(chain.id)
                resseq = residue.id[1]
                icode = residue.id[2].strip()

                # skip insertion-coded residues for now (100J etc.)
                if icode:
                    continue

                key = (chain_id, resseq)
                if key in seen:
                    continue
                seen.add(key)

                wt = AA3_TO_1[resname]
                example_mut = f"{wt}{chain_id}{resseq}V" if wt != "V" else f"{wt}{chain_id}{resseq}A"

                rows.append({
                    "wt_aa": wt,
                    "chain": chain_id,
                    "position": resseq,
                    "example_mutation": example_mut,
                })

    df = pd.DataFrame(rows).sort_values(["chain", "position"]).reset_index(drop=True)
    return df


def foldx_scan_available_mutations(pdb_mode, uploaded_pdb, saved_pdb):
    pdb_path = choose_pdb(pdb_mode, uploaded_pdb, saved_pdb)
    df = scan_pdb_mutations(pdb_path)
    preview = df.head(50)
    summary = f"Found {len(df)} valid standard residue sites in {pdb_path.name}. Showing first 50."
    return summary, preview


# -------------------------
# plotting helpers
# -------------------------
def short_summary_text(summary: dict, tool: str, mutation_note: str = ""):
    lines = [f"Tool: {tool}"]
    if "sample_name" in summary:
        lines.append(f"Sample: {summary['sample_name']}")
    if "sequence_length" in summary:
        lines.append(f"Sequence length: {summary['sequence_length']}")
    if mutation_note:
        lines.append(mutation_note)

    if tool == "AntiBERTy":
        lines.append(f"Embedding shape: {summary.get('residue_embedding_shape', summary.get('embedding_shape'))}")
        lines.append("Main interpretation: residue-level antibody embedding signal.")
    elif tool == "ESM-2":
        lines.append(f"Embedding shape: {summary.get('embedding_shape')}")
        lines.append("Main interpretation: residue-level protein language-model signal.")
    elif tool == "Antibody-Evolution":
        lines.append(f"Mutations scored: {summary.get('num_mutations')}")
        lines.append(f"Models used: {', '.join(summary.get('models_used', []))}")
        lines.append("Main interpretation: mutation landscape and sensitive positions.")
    elif tool == "FoldX":
        lines.append(f"Mutation: {summary.get('mutation')}")
        lines.append(f"Repaired PDB: {summary.get('repaired_pdb')}")
        lines.append("Main interpretation: structure-based energy terms for the selected mutation.")

    return "\n".join(lines)


def make_line_plot(df, x_col, y_col, title, mutation_pos=None, mutant_df=None, mutant_label="Mutant"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x_col], df[y_col], label="WT", linewidth=2)
    if mutant_df is not None:
        ax.plot(mutant_df[x_col], mutant_df[y_col], label=mutant_label, linewidth=2, alpha=0.8)
        ax.legend()
    if mutation_pos is not None:
        ax.axvline(mutation_pos, linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_bar_plot(df, x_col, y_col, title, top_n=15, highlight_value=None):
    plot_df = df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["C0"] * len(plot_df)
    if highlight_value is not None and x_col in plot_df.columns:
        for i, val in enumerate(plot_df[x_col].astype(str)):
            if str(val) == str(highlight_value):
                colors[i] = "C3"
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color=colors)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    return fig


def make_heatmap(df_values, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(df_values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout()
    return fig


def make_umap_plot(embeddings, residues, title):
    reducer = umap.UMAP(random_state=42, n_neighbors=min(15, max(2, len(embeddings)-1)))
    emb2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=np.arange(len(embeddings)), s=25)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # label a few residues to keep readable
    step = max(1, len(residues) // 12)
    for i in range(0, len(residues), step):
        ax.text(emb2d[i, 0], emb2d[i, 1], f"{residues[i]}{i+1}", fontsize=8)

    plt.tight_layout()
    return fig


def top_norm_table(csv_path: Path, n=10):
    df = pd.read_csv(csv_path)
    return df.sort_values("embedding_norm", ascending=False).head(n)


def abevo_heatmap_from_mutations(df_mut):
    letters = list("ACDEFGHIKLMNPQRSTVWY")
    df = df_mut.copy()

    def pos_of(mut):
        m = re.search(r"(\d+)", mut)
        return int(m.group(1)) if m else -1

    def mut_aa(mut):
        return mut[-1]

    df["position"] = df["mutant"].apply(pos_of)
    df["mut_aa"] = df["mutant"].apply(mut_aa)

    pivot = df.pivot_table(index="mut_aa", columns="position", values="score_mean", aggfunc="mean")
    pivot = pivot.reindex(index=letters)
    return pivot


# -------------------------
# PDF export
# -------------------------
def save_pdf_report(tool, summary_text, main_df, extra_df, fig):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_DIR / f"{safe_name(tool)}_report_{ts}.pdf"

    with PdfPages(pdf_path) as pdf:
        fig1, ax1 = plt.subplots(figsize=(11, 8.5))
        ax1.axis("off")
        ax1.text(
            0.02, 0.98,
            f"{tool} Report\n\n{summary_text}",
            va="top",
            ha="left",
            fontsize=12,
            family="monospace"
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight")

        if main_df is not None and len(main_df) > 0:
            fig2, ax2 = plt.subplots(figsize=(11, 8.5))
            ax2.axis("off")
            table = ax2.table(
                cellText=main_df.head(20).values,
                colLabels=list(main_df.columns),
                loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.4)
            ax2.set_title("Main Table", pad=20)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        if extra_df is not None and len(extra_df) > 0:
            fig3, ax3 = plt.subplots(figsize=(11, 8.5))
            ax3.axis("off")
            table = ax3.table(
                cellText=extra_df.head(25).values,
                colLabels=list(extra_df.columns),
                loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.4)
            ax3.set_title("Extra Details", pad=20)
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

    return str(pdf_path)


# -------------------------
# tool execution
# -------------------------
def run_antiberty_from_app(input_mode, pasted_sequence, saved_fasta, sequence_mutation):
    fasta_path, stem, wt_seq = choose_sequence(input_mode, pasted_sequence, saved_fasta)

    runner = runner_path("run_antiberty_3.py")

    cmd = [
        "conda", "run", "-n", "antiberty",
        "python", str(runner),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = RESULTS_DIR / f"{stem}_antiberty"
    if input_mode == "Paste sequence":
        result_dir = RESULTS_DIR / "input_sequence_antiberty"
    if not result_dir.exists():
        raise RuntimeError(f"AntiBERTy result folder not found: {result_dir}")

    summary = load_json(result_dir / "summary.json")
    wt_df = pd.read_csv(result_dir / "residue_norms.csv")
    wt_embeddings = np.load(result_dir / "residue_embeddings.npy")

    mutation_note = ""
    mutant_df = None
    if sequence_mutation and sequence_mutation.strip():
        mutated_seq = apply_seq_mutation(wt_seq, sequence_mutation.strip())
        mut_fasta = TMP_DIR / "input_mutant_sequence.fasta"
        write_fasta(mutated_seq, mut_fasta)

        cmd2 = [
            "conda", "run", "-n", "antiberty",
            "python", str(runner),
            str(mut_fasta),
        ]
        stdout2, stderr2, code2 = run_subprocess(cmd2)
        if code2 != 0:
            raise RuntimeError(stderr2 or stdout2)

        mut_result_dir = RESULTS_DIR / "input_mutant_sequence_antiberty"
        if mut_result_dir.exists():
            mutant_df = pd.read_csv(mut_result_dir / "residue_norms.csv")
        mutation_note = f"Compared WT against sequence mutation: {sequence_mutation.strip()}"

    main_table = wt_df.sort_values("embedding_norm", ascending=False).head(10)
    mut_pos = parse_seq_mutation(sequence_mutation)[1] if sequence_mutation and sequence_mutation.strip() else None
    plot = make_line_plot(
        wt_df, "position", "embedding_norm",
        "AntiBERTy Residue Embedding Norms",
        mutation_pos=mut_pos,
        mutant_df=mutant_df,
        mutant_label="Mutant"
    )

    # Extra details = full residue table sorted by importance
    extra_df = wt_df.sort_values("embedding_norm", ascending=False).reset_index(drop=True)

    summary_text = short_summary_text(summary, "AntiBERTy", mutation_note)
    pdf_path = save_pdf_report("AntiBERTy", summary_text, main_table, extra_df, plot)

    # UMAP + heatmap additional plot selection is handled separately in app
    return summary_text, main_table, plot, extra_df, json.dumps(summary, indent=2), pdf_path


def run_esm2_from_app(input_mode, pasted_sequence, saved_fasta, sequence_mutation):
    fasta_path, stem, wt_seq = choose_sequence(input_mode, pasted_sequence, saved_fasta)

    runner = runner_path("run_esm2_3.py")

    cmd = [
        "conda", "run", "-n", "esm2",
        "python", str(runner),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = RESULTS_DIR / f"{stem}_esm2"
    if input_mode == "Paste sequence":
        result_dir = RESULTS_DIR / "input_sequence_esm2"
    if not result_dir.exists():
        raise RuntimeError(f"ESM-2 result folder not found: {result_dir}")

    summary = load_json(result_dir / "summary.json")
    wt_df = pd.read_csv(result_dir / "residue_norms.csv")

    mutation_note = ""
    mutant_df = None
    if sequence_mutation and sequence_mutation.strip():
        mutated_seq = apply_seq_mutation(wt_seq, sequence_mutation.strip())
        mut_fasta = TMP_DIR / "input_mutant_sequence.fasta"
        write_fasta(mutated_seq, mut_fasta)

        cmd2 = [
            "conda", "run", "-n", "esm2",
            "python", str(runner),
            str(mut_fasta),
        ]
        stdout2, stderr2, code2 = run_subprocess(cmd2)
        if code2 != 0:
            raise RuntimeError(stderr2 or stdout2)

        mut_result_dir = RESULTS_DIR / "input_mutant_sequence_esm2"
        if mut_result_dir.exists():
            mutant_df = pd.read_csv(mut_result_dir / "residue_norms.csv")
        mutation_note = f"Compared WT against sequence mutation: {sequence_mutation.strip()}"

    main_table = wt_df.sort_values("embedding_norm", ascending=False).head(10)
    mut_pos = parse_seq_mutation(sequence_mutation)[1] if sequence_mutation and sequence_mutation.strip() else None
    plot = make_line_plot(
        wt_df, "position", "embedding_norm",
        "ESM-2 Residue Embedding Norms",
        mutation_pos=mut_pos,
        mutant_df=mutant_df,
        mutant_label="Mutant"
    )
    extra_df = wt_df.sort_values("embedding_norm", ascending=False).reset_index(drop=True)

    summary_text = short_summary_text(summary, "ESM-2", mutation_note)
    pdf_path = save_pdf_report("ESM-2", summary_text, main_table, extra_df, plot)

    return summary_text, main_table, plot, extra_df, json.dumps(summary, indent=2), pdf_path


def run_abevo_from_app(input_mode, pasted_sequence, saved_fasta, sequence_mutation):
    fasta_path, stem, wt_seq = choose_sequence(input_mode, pasted_sequence, saved_fasta)

    runner = runner_path("run_abevo_2.py")

    cmd = [
        "conda", "run", "-n", "abevo",
        "python", str(runner),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = RESULTS_DIR / f"{stem}_abevo"
    if input_mode == "Paste sequence":
        result_dir = RESULTS_DIR / "input_sequence_abevo"
    if not result_dir.exists():
        raise RuntimeError(f"Antibody-Evolution result folder not found: {result_dir}")

    summary = load_json(result_dir / "summary.json")
    top_df = pd.read_csv(result_dir / "top_mutations.csv")
    bottom_df = pd.read_csv(result_dir / "bottom_mutations.csv")
    all_df = pd.read_csv(result_dir / "mutations.csv")

    mutation_note = ""
    main_table = top_df.head(10)
    extra_df = bottom_df.head(10)
    highlight_mut = None

    if sequence_mutation and sequence_mutation.strip():
        seq_mut = sequence_mutation.strip().replace(";", "")
        hit = all_df[all_df["mutant"] == seq_mut]
        if not hit.empty:
            main_table = hit[["mutant", "score_mean"] + [c for c in hit.columns if c.startswith("esm")]].head(1)
            mutation_note = f"Highlighted mutation: {seq_mut}"
            highlight_mut = seq_mut
        else:
            mutation_note = f"Mutation {seq_mut} not found in score table; showing top mutations."

    plot = make_bar_plot(
        top_df,
        "mutant",
        "score_mean",
        "Top Antibody-Evolution Mutations",
        top_n=15,
        highlight_value=highlight_mut
    )

    summary_text = short_summary_text(summary, "Antibody-Evolution", mutation_note)
    pdf_path = save_pdf_report("Antibody-Evolution", summary_text, main_table, extra_df, plot)

    return summary_text, main_table, plot, extra_df, json.dumps(summary, indent=2), pdf_path


def run_foldx_from_app(pdb_mode, uploaded_pdb, saved_pdb, foldx_mutation):
    pdb_path = choose_pdb(pdb_mode, uploaded_pdb, saved_pdb)

    if not foldx_mutation or not foldx_mutation.strip():
        raise ValueError("Please provide a FoldX mutation like AB114V")

    foldx_mutation = foldx_mutation.strip().replace(";", "")

    runner = runner_path("run_foldx.py")

    cmd = [
        "python",
        str(runner),
        str(pdb_path),
        foldx_mutation,
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = RESULTS_DIR / f"{pdb_path.stem}_foldx_{foldx_mutation}"
    if not result_dir.exists():
        raise RuntimeError(f"FoldX result folder not found: {result_dir}")

    summary = load_json(result_dir / "summary.json")
    energy_df = pd.read_csv(result_dir / "energy_terms.csv")

    important_terms = [
        "BackHbond", "SideHbond", "Energy_VdW", "Electro",
        "Energy_SolvP", "Energy_SolvH", "Energy_vdwclash", "Total"
    ]
    main_table = energy_df[energy_df["term"].isin(important_terms)].copy()
    extra_df = energy_df

    plot = make_bar_plot(main_table, "term", "value", "FoldX Key Energy Terms", top_n=20)

    summary_text = short_summary_text(summary, "FoldX", f"Analyzed mutation: {foldx_mutation}")
    pdf_path = save_pdf_report("FoldX", summary_text, main_table, extra_df, plot)

    return summary_text, main_table, plot, extra_df, json.dumps(summary, indent=2), pdf_path


def generate_extra_plot(tool, seq_input_mode, pasted_sequence, saved_fasta,
                        pdb_input_mode, uploaded_pdb, saved_pdb, plot_choice):
    """
    Separate helper for richer plots:
    - AntiBERTy: UMAP, heatmap
    - ESM-2: UMAP, heatmap
    - ABEVO: heatmap
    - FoldX: mutation availability table already handled separately
    """
    if tool in ["AntiBERTy", "ESM-2"]:
        fasta_path, stem, _ = choose_sequence(seq_input_mode, pasted_sequence, saved_fasta)
        suffix = "antiberty" if tool == "AntiBERTy" else "esm2"
        result_dir = RESULTS_DIR / f"{stem}_{suffix}"
        if seq_input_mode == "Paste sequence":
            result_dir = RESULTS_DIR / f"input_sequence_{suffix}"

        summary = load_json(result_dir / "summary.json")
        emb = np.load(result_dir / "residue_embeddings.npy")
        residue_df = pd.read_csv(result_dir / "residue_norms.csv")
        residues = residue_df["residue"].tolist()

        if plot_choice == "UMAP":
            fig = make_umap_plot(emb, residues, f"{tool} Residue UMAP")
            return fig
        elif plot_choice == "Heatmap":
            # first 30 dims for readability
            fig = make_heatmap(emb[:, :30].T, f"{tool} Embedding Heatmap (first 30 dims)", "Residue position", "Embedding dim")
            return fig

    elif tool == "Antibody-Evolution":
        fasta_path, stem, _ = choose_sequence(seq_input_mode, pasted_sequence, saved_fasta)
        result_dir = RESULTS_DIR / f"{stem}_abevo"
        if seq_input_mode == "Paste sequence":
            result_dir = RESULTS_DIR / "input_sequence_abevo"

        all_df = pd.read_csv(result_dir / "mutations.csv")
        pivot = abevo_heatmap_from_mutations(all_df)
        fig = make_heatmap(pivot.values, "Antibody-Evolution Mutation Heatmap", "Sequence position", "Mutant amino acid")
        return fig

    # fallback blank
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, "No extra plot available for this combination.", ha="center", va="center")
    return fig


def main_run(tool, seq_input_mode, pasted_sequence, saved_fasta, seq_mutation,
             pdb_input_mode, uploaded_pdb, saved_pdb, foldx_mutation):
    if tool == "AntiBERTy":
        return run_antiberty_from_app(seq_input_mode, pasted_sequence, saved_fasta, seq_mutation)
    elif tool == "ESM-2":
        return run_esm2_from_app(seq_input_mode, pasted_sequence, saved_fasta, seq_mutation)
    elif tool == "Antibody-Evolution":
        return run_abevo_from_app(seq_input_mode, pasted_sequence, saved_fasta, seq_mutation)
    elif tool == "FoldX":
        return run_foldx_from_app(pdb_input_mode, uploaded_pdb, saved_pdb, foldx_mutation)
    else:
        raise ValueError("Unknown tool selected.")


def toggle_inputs(tool, seq_mode, pdb_mode):
    seq_visible = tool in ["AntiBERTy", "ESM-2", "Antibody-Evolution"]
    foldx_visible = tool == "FoldX"

    return (
        gr.update(visible=seq_visible),
        gr.update(visible=seq_visible and seq_mode == "Paste sequence"),
        gr.update(visible=seq_visible and seq_mode == "Choose saved FASTA"),
        gr.update(visible=seq_visible),
        gr.update(visible=foldx_visible),
        gr.update(visible=foldx_visible and pdb_mode == "Upload PDB"),
        gr.update(visible=foldx_visible and pdb_mode == "Choose saved PDB"),
        gr.update(visible=foldx_visible),
        gr.update(visible=tool in ["AntiBERTy", "ESM-2", "Antibody-Evolution"]),
        gr.update(visible=tool == "FoldX"),
    )


saved_fastas = list_saved_fastas()
saved_pdbs = list_saved_pdbs()

with gr.Blocks(title="Antibody Benchmark GUI") as demo:
    gr.Markdown("# Antibody Benchmark GUI")
    gr.Markdown("Run AntiBERTy, ESM-2, Antibody-Evolution, and FoldX with mentor-friendly outputs.")

    with gr.Row():
        with gr.Column(scale=1):
            tool = gr.Dropdown(
                ["AntiBERTy", "ESM-2", "Antibody-Evolution", "FoldX"],
                value="AntiBERTy",
                label="Select Tool"
            )

            with gr.Group(visible=True) as seq_group:
                seq_input_mode = gr.Radio(
                    ["Paste sequence", "Choose saved FASTA"],
                    value="Choose saved FASTA",
                    label="Sequence Input Mode"
                )
                pasted_sequence = gr.Textbox(
                    label="Paste Sequence",
                    lines=8,
                    placeholder="Paste one protein/antibody chain sequence here",
                    visible=False
                )
                saved_fasta = gr.Dropdown(
                    choices=saved_fastas,
                    value=saved_fastas[0] if saved_fastas and saved_fastas[0] else None,
                    label="Choose Saved FASTA",
                    visible=True
                )
                seq_mutation = gr.Textbox(
                    label="Optional Sequence Mutation",
                    placeholder="Example: A23V",
                    visible=True
                )

                extra_plot_choice = gr.Radio(
                    ["UMAP", "Heatmap"],
                    value="UMAP",
                    label="Extra Plot Type",
                    visible=True
                )
                extra_plot_btn = gr.Button("Generate Extra Plot", visible=True)

            with gr.Group(visible=False) as pdb_group:
                pdb_input_mode = gr.Radio(
                    ["Upload PDB", "Choose saved PDB"],
                    value="Choose saved PDB",
                    label="PDB Input Mode"
                )
                uploaded_pdb = gr.File(label="Upload PDB", type="filepath", visible=False)
                saved_pdb = gr.Dropdown(
                    choices=saved_pdbs,
                    value=saved_pdbs[0] if saved_pdbs and saved_pdbs[0] else None,
                    label="Choose Saved PDB",
                    visible=True
                )
                foldx_mutation = gr.Textbox(
                    label="FoldX Mutation",
                    placeholder="Example: AB114V",
                    visible=True
                )
                scan_btn = gr.Button("Scan Valid FoldX Mutations", visible=True)

            run_btn = gr.Button("Run Analysis")
            report_file = gr.File(label="Download PDF Report")

        with gr.Column(scale=2):
            with gr.Tab("Important Summary"):
                summary_text = gr.Textbox(label="Key Findings", lines=9)

            with gr.Tab("Main Table"):
                main_table = gr.Dataframe(label="Important Result Table")

            with gr.Tab("Main Plot"):
                plot = gr.Plot(label="Main Visualization")

            with gr.Tab("Extra Details"):
                extra_table = gr.Dataframe(label="Additional Details")

            with gr.Tab("Extra Plot"):
                extra_plot = gr.Plot(label="Advanced Visualization")

            with gr.Tab("FoldX Valid Mutations"):
                foldx_scan_summary = gr.Textbox(label="Scan Summary", lines=3)
                foldx_scan_table = gr.Dataframe(label="Available Mutation Sites")

            with gr.Tab("Raw JSON"):
                raw_json = gr.Code(label="Saved Summary JSON", language="json")

    tool.change(
        toggle_inputs,
        inputs=[tool, seq_input_mode, pdb_input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, seq_mutation,
                 pdb_group, uploaded_pdb, saved_pdb, foldx_mutation,
                 extra_plot_choice, scan_btn],
    )

    seq_input_mode.change(
        toggle_inputs,
        inputs=[tool, seq_input_mode, pdb_input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, seq_mutation,
                 pdb_group, uploaded_pdb, saved_pdb, foldx_mutation,
                 extra_plot_choice, scan_btn],
    )

    pdb_input_mode.change(
        toggle_inputs,
        inputs=[tool, seq_input_mode, pdb_input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, seq_mutation,
                 pdb_group, uploaded_pdb, saved_pdb, foldx_mutation,
                 extra_plot_choice, scan_btn],
    )

    run_btn.click(
        main_run,
        inputs=[tool, seq_input_mode, pasted_sequence, saved_fasta, seq_mutation,
                pdb_input_mode, uploaded_pdb, saved_pdb, foldx_mutation],
        outputs=[summary_text, main_table, plot, extra_table, raw_json, report_file],
    )

    extra_plot_btn.click(
        generate_extra_plot,
        inputs=[tool, seq_input_mode, pasted_sequence, saved_fasta,
                pdb_input_mode, uploaded_pdb, saved_pdb, extra_plot_choice],
        outputs=[extra_plot],
    )

    scan_btn.click(
        foldx_scan_available_mutations,
        inputs=[pdb_input_mode, uploaded_pdb, saved_pdb],
        outputs=[foldx_scan_summary, foldx_scan_table],
    )

demo.launch(
    allowed_paths=[
        str(BASE_DIR),
        str(REPORT_DIR),
        str(RESULTS_DIR),
    ]
)
