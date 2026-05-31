import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
RUNNERS_DIR = BASE_DIR / "runners"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TMP_DIR = BASE_DIR / "tmp"
REPORT_DIR = RESULTS_DIR / "report_exports"

TMP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# helpers
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


def latest_result_dir(prefix: str):
    if not RESULTS_DIR.exists():
        return None
    matches = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


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


def parse_seq_mutation(mut: str):
    """
    A23V
    """
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
    mutated = seq[:idx] + mt + seq[idx + 1 :]
    return mutated


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


def top_norm_table(csv_path: Path, n=10):
    df = pd.read_csv(csv_path)
    return df.sort_values("embedding_norm", ascending=False).head(n)


def save_pdf_report(tool, summary_text, main_df, extra_df, fig):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_DIR / f"{safe_name(tool)}_report_{ts}.pdf"

    with PdfPages(pdf_path) as pdf:
        # Page 1: summary
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

        # Page 2: main plot
        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight")

        # Page 3: main table
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

        # Page 4: extra table
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
# tool logic
# -------------------------
def run_antiberty_from_app(input_mode, pasted_sequence, saved_fasta, sequence_mutation):
    fasta_path, stem, wt_seq = choose_sequence(input_mode, pasted_sequence, saved_fasta)

    runner = runner_path("run_antiberty_2.py")

    # WT
    cmd = [
        "conda", "run", "-n", "antiberty",
        "python", str(runner),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = RESULTS_DIR / f"{stem}_antiberty"
    if not result_dir.exists():
        raise RuntimeError(f"AntiBERTy result folder not found: {result_dir}")

    summary = load_json(result_dir / "summary.json")
    wt_df = pd.read_csv(result_dir / "residue_norms.csv")

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
    extra_df = wt_df

    summary_text = short_summary_text(summary, "AntiBERTy", mutation_note)
    pdf_path = save_pdf_report("AntiBERTy", summary_text, main_table, extra_df, plot)

    return summary_text, main_table, plot, extra_df, json.dumps(summary, indent=2), pdf_path


def run_esm2_from_app(input_mode, pasted_sequence, saved_fasta, sequence_mutation):
    fasta_path, stem, wt_seq = choose_sequence(input_mode, pasted_sequence, saved_fasta)

    runner = runner_path("run_esm2_2.py")

    cmd = [
        "conda", "run", "-n", "esm2",
        "python", str(runner),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = RESULTS_DIR / f"{stem}_esm2"
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
    extra_df = wt_df

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
        gr.update(visible=seq_visible),                          # seq group
        gr.update(visible=seq_visible and seq_mode == "Paste sequence"),      # paste seq
        gr.update(visible=seq_visible and seq_mode == "Choose saved FASTA"),  # saved fasta
        gr.update(visible=seq_visible),                          # seq mutation
        gr.update(visible=foldx_visible),                        # pdb group
        gr.update(visible=foldx_visible and pdb_mode == "Upload PDB"),        # upload pdb
        gr.update(visible=foldx_visible and pdb_mode == "Choose saved PDB"),  # saved pdb
        gr.update(visible=foldx_visible),                        # foldx mutation
    )


saved_fastas = list_saved_fastas()
saved_pdbs = list_saved_pdbs()

with gr.Blocks(title="Antibody Benchmark GUI") as demo:
    gr.Markdown("# Antibody Benchmark GUI")
    gr.Markdown("Run AntiBERTy, ESM-2, Antibody-Evolution, and FoldX with user-friendly outputs.")

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

            with gr.Tab("Raw JSON"):
                raw_json = gr.Code(label="Saved Summary JSON", language="json")

    tool.change(
        toggle_inputs,
        inputs=[tool, seq_input_mode, pdb_input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, seq_mutation,
                 pdb_group, uploaded_pdb, saved_pdb, foldx_mutation],
    )

    seq_input_mode.change(
        toggle_inputs,
        inputs=[tool, seq_input_mode, pdb_input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, seq_mutation,
                 pdb_group, uploaded_pdb, saved_pdb, foldx_mutation],
    )

    pdb_input_mode.change(
        toggle_inputs,
        inputs=[tool, seq_input_mode, pdb_input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, seq_mutation,
                 pdb_group, uploaded_pdb, saved_pdb, foldx_mutation],
    )

    run_btn.click(
        main_run,
        inputs=[tool, seq_input_mode, pasted_sequence, saved_fasta, seq_mutation,
                pdb_input_mode, uploaded_pdb, saved_pdb, foldx_mutation],
        outputs=[summary_text, main_table, plot, extra_table, raw_json, report_file],
    )

demo.launch(
    allowed_paths=[
        str(BASE_DIR),
        str(REPORT_DIR),
        str(RESULTS_DIR),
    ]
)
