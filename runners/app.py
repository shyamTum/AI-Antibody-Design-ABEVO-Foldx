import json
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
RUNNERS_DIR = BASE_DIR / "runners"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


# ----------------------------
# Helpers
# ----------------------------
def list_saved_fastas():
    fastas = sorted(DATA_DIR.rglob("*.fasta"))
    rels = [str(f.relative_to(BASE_DIR)) for f in fastas]
    return rels if rels else [""]


def save_temp_fasta(seq: str) -> Path:
    seq = seq.strip().replace(" ", "").replace("\n", "")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w") as f:
        f.write(">input_sequence\n")
        f.write(seq + "\n")
        return Path(f.name)


def latest_result_dir(prefix: str):
    if not RESULTS_DIR.exists():
        return None
    candidates = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_subprocess(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def short_summary_text(summary: dict, tool: str) -> str:
    if tool == "AntiBERTy":
        return (
            f"Tool: AntiBERTy\n"
            f"Sample: {summary.get('sample_name')}\n"
            f"Sequence length: {summary.get('sequence_length')}\n"
            f"Residue embedding shape: {summary.get('residue_embedding_shape', summary.get('embedding_shape'))}\n"
            f"Main use: residue-level antibody embedding profile"
        )
    elif tool == "ESM-2":
        return (
            f"Tool: ESM-2\n"
            f"Sample: {summary.get('sample_name')}\n"
            f"Sequence length: {summary.get('sequence_length')}\n"
            f"Residue embedding shape: {summary.get('embedding_shape')}\n"
            f"Main use: general protein language-model representation"
        )
    elif tool == "Antibody-Evolution":
        return (
            f"Tool: Antibody-Evolution\n"
            f"Sample: {summary.get('sample_name')}\n"
            f"Sequence length: {summary.get('sequence_length')}\n"
            f"Number of mutations scored: {summary.get('num_mutations')}\n"
            f"Models used: {', '.join(summary.get('models_used', []))}\n"
            f"Main use: mutation landscape and position sensitivity"
        )
    elif tool == "FoldX":
        return (
            f"Tool: FoldX\n"
            f"Sample: {summary.get('sample_name')}\n"
            f"Mutation: {summary.get('mutation')}\n"
            f"Repaired PDB: {summary.get('repaired_pdb')}\n"
            f"Main use: structure-based mutation energy analysis"
        )
    return json.dumps(summary, indent=2)


def make_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, top_n=20):
    df = df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    return fig


def choose_sequence(mode, pasted_sequence, saved_fasta_relpath):
    if mode == "Paste sequence":
        seq = (pasted_sequence or "").strip()
        if not seq:
            raise ValueError("Please paste a sequence.")
        return save_temp_fasta(seq), "input_sequence"
    else:
        if not saved_fasta_relpath:
            raise ValueError("Please choose a saved FASTA.")
        fasta_path = BASE_DIR / saved_fasta_relpath
        if not fasta_path.exists():
            raise FileNotFoundError(f"Saved FASTA not found: {fasta_path}")
        return fasta_path, Path(saved_fasta_relpath).stem


def top_norm_table(csv_path: Path, n=10):
    df = pd.read_csv(csv_path)
    return df.sort_values("embedding_norm", ascending=False).head(n)


# ----------------------------
# Tool runners
# ----------------------------
def run_antiberty_from_app(mode, pasted_sequence, saved_fasta):
    fasta_path, stem = choose_sequence(mode, pasted_sequence, saved_fasta)

    cmd = [
        "conda", "run", "-n", "antiberty",
        "python", str(RUNNERS_DIR / "run_antiberty_2.py"),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = latest_result_dir(f"{stem}_antiberty")
    if result_dir is None:
        # for pasted sequence
        result_dir = latest_result_dir("input_sequence_antiberty")
    if result_dir is None:
        raise RuntimeError("AntiBERTy result folder not found.")

    summary = load_json(result_dir / "summary.json")
    norms_df = pd.read_csv(result_dir / "residue_norms.csv")
    main_table = top_norm_table(result_dir / "residue_norms.csv", n=10)
    plot = make_line_plot(norms_df, "position", "embedding_norm", "AntiBERTy Residue Embedding Norms")

    return (
        short_summary_text(summary, "AntiBERTy"),
        main_table,
        plot,
        norms_df.head(25),
        json.dumps(summary, indent=2),
    )


def run_esm2_from_app(mode, pasted_sequence, saved_fasta):
    fasta_path, stem = choose_sequence(mode, pasted_sequence, saved_fasta)

    cmd = [
        "conda", "run", "-n", "esm2",
        "python", str(RUNNERS_DIR / "run_esm2_2.py"),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = latest_result_dir(f"{stem}_esm2")
    if result_dir is None:
        result_dir = latest_result_dir("input_sequence_esm2")
    if result_dir is None:
        raise RuntimeError("ESM-2 result folder not found.")

    summary = load_json(result_dir / "summary.json")
    norms_df = pd.read_csv(result_dir / "residue_norms.csv")
    main_table = top_norm_table(result_dir / "residue_norms.csv", n=10)
    plot = make_line_plot(norms_df, "position", "embedding_norm", "ESM-2 Residue Embedding Norms")

    return (
        short_summary_text(summary, "ESM-2"),
        main_table,
        plot,
        norms_df.head(25),
        json.dumps(summary, indent=2),
    )


def run_abevo_from_app(mode, pasted_sequence, saved_fasta):
    fasta_path, stem = choose_sequence(mode, pasted_sequence, saved_fasta)

    cmd = [
        "conda", "run", "-n", "abevo",
        "python", str(RUNNERS_DIR / "run_abevo_2.py"),
        str(fasta_path),
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = latest_result_dir(f"{stem}_abevo")
    if result_dir is None:
        result_dir = latest_result_dir("input_sequence_abevo")
    if result_dir is None:
        raise RuntimeError("Antibody-Evolution result folder not found.")

    summary = load_json(result_dir / "summary.json")
    top_df = pd.read_csv(result_dir / "top_mutations.csv")
    bottom_df = pd.read_csv(result_dir / "bottom_mutations.csv")
    pos_df = pd.read_csv(result_dir / "position_scores.csv")

    main_table = top_df.head(10)
    plot = make_bar_plot(top_df, "mutant", "score_mean", "Top Antibody-Evolution Mutations", top_n=15)
    secondary = bottom_df.head(10)

    return (
        short_summary_text(summary, "Antibody-Evolution"),
        main_table,
        plot,
        secondary,
        json.dumps(summary, indent=2),
    )


def run_foldx_from_app(pdb_file, mutation):
    if pdb_file is None:
        raise ValueError("Please upload a PDB for FoldX.")
    if not mutation or not mutation.strip():
        raise ValueError("Please provide a valid FoldX mutation, e.g. AB114V")

    pdb_path = Path(pdb_file)
    mutation = mutation.strip().replace(";", "")

    cmd = [
        "python",
        str(RUNNERS_DIR / "run_foldx.py"),
        str(pdb_path),
        mutation,
    ]
    stdout, stderr, code = run_subprocess(cmd)
    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = latest_result_dir(f"{pdb_path.stem}_foldx_{mutation}")
    if result_dir is None:
        raise RuntimeError("FoldX result folder not found.")

    summary = load_json(result_dir / "summary.json")
    energy_df = pd.read_csv(result_dir / "energy_terms.csv")

    # prioritize only the most important energy terms
    important_terms = [
        "BackHbond", "SideHbond", "Energy_VdW", "Electro",
        "Energy_SolvP", "Energy_SolvH", "Energy_vdwclash", "Total"
    ]
    main_table = energy_df[energy_df["term"].isin(important_terms)].copy()
    plot = make_bar_plot(main_table, "term", "value", "FoldX Key Energy Terms", top_n=20)

    return (
        short_summary_text(summary, "FoldX"),
        main_table,
        plot,
        energy_df,
        json.dumps(summary, indent=2),
    )


def main_run(tool, input_mode, pasted_sequence, saved_fasta, pdb_file, mutation):
    if tool == "AntiBERTy":
        return run_antiberty_from_app(input_mode, pasted_sequence, saved_fasta)
    elif tool == "ESM-2":
        return run_esm2_from_app(input_mode, pasted_sequence, saved_fasta)
    elif tool == "Antibody-Evolution":
        return run_abevo_from_app(input_mode, pasted_sequence, saved_fasta)
    elif tool == "FoldX":
        return run_foldx_from_app(pdb_file, mutation)
    else:
        raise ValueError("Unknown tool selected.")


# ----------------------------
# UI behavior
# ----------------------------
def toggle_inputs(tool, input_mode):
    show_sequence_inputs = tool in ["AntiBERTy", "ESM-2", "Antibody-Evolution"]
    show_foldx_inputs = tool == "FoldX"

    return (
        gr.update(visible=show_sequence_inputs),  # sequence input group
        gr.update(visible=(show_sequence_inputs and input_mode == "Paste sequence")),  # pasted sequence
        gr.update(visible=(show_sequence_inputs and input_mode == "Choose saved FASTA")),  # saved fasta
        gr.update(visible=show_foldx_inputs),  # foldx group
    )


# ----------------------------
# App
# ----------------------------
saved_fastas = list_saved_fastas()

with gr.Blocks(title="Antibody Benchmark GUI") as demo:
    gr.Markdown("# Antibody Benchmark GUI")
    gr.Markdown(
        "Run AntiBERTy, ESM-2, Antibody-Evolution, and FoldX with user-friendly outputs."
    )

    with gr.Row():
        with gr.Column(scale=1):
            tool = gr.Dropdown(
                ["AntiBERTy", "ESM-2", "Antibody-Evolution", "FoldX"],
                value="AntiBERTy",
                label="Select Tool",
            )

            input_mode = gr.Radio(
                ["Paste sequence", "Choose saved FASTA"],
                value="Paste sequence",
                label="Sequence Input Mode",
            )

            with gr.Group(visible=True) as seq_group:
                pasted_sequence = gr.Textbox(
                    label="Paste Sequence",
                    lines=8,
                    placeholder="Paste one protein/antibody chain sequence here",
                    visible=True,
                )
                saved_fasta = gr.Dropdown(
                    choices=saved_fastas,
                    value=saved_fastas[0] if saved_fastas and saved_fastas[0] else None,
                    label="Choose Saved FASTA",
                    visible=False,
                )

            with gr.Group(visible=False) as foldx_group:
                pdb_file = gr.File(label="Upload PDB for FoldX")
                mutation = gr.Textbox(label="FoldX Mutation", placeholder="Example: AB114V")

            run_btn = gr.Button("Run Analysis")

        with gr.Column(scale=2):
            with gr.Tab("Important Summary"):
                summary_text = gr.Textbox(label="Key Findings", lines=8)

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
        inputs=[tool, input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, foldx_group],
    )

    input_mode.change(
        toggle_inputs,
        inputs=[tool, input_mode],
        outputs=[seq_group, pasted_sequence, saved_fasta, foldx_group],
    )

    run_btn.click(
        main_run,
        inputs=[tool, input_mode, pasted_sequence, saved_fasta, pdb_file, mutation],
        outputs=[summary_text, main_table, plot, extra_table, raw_json],
    )

demo.launch()
