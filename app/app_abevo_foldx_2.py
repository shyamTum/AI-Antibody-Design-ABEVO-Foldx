import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser

BASE_DIR = Path("/home/ghoshlab/Desktop/Shyam/antibody_benchmark")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RUNNERS_DIR = BASE_DIR / "runners"
REPORT_DIR = RESULTS_DIR / "report_exports"
STRUCTURE_DIR = RESULTS_DIR / "structure_figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def list_fastas():
    fastas = []
    for p in sorted(DATA_DIR.rglob("*.fasta")):
        if "_chain_" in p.name:
            fastas.append(str(p.relative_to(BASE_DIR)))
    return fastas or [""]


def list_pdbs():
    return [str(p.relative_to(BASE_DIR)) for p in sorted(DATA_DIR.rglob("*.pdb"))] or [""]


def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line)
    return "".join(seq)


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def get_result_dir(fasta_path, pdb_path, chain_id):
    return RESULTS_DIR / safe_name(f"{fasta_path.stem}_{pdb_path.stem}_chain_{chain_id}_PIPELINE_V2")


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
                    "chain": str(chain.id),
                    "position": resseq,
                    "example_foldx_mutation": example,
                })

    df = pd.DataFrame(rows).sort_values(["chain", "position"])
    return f"Found {len(df)} valid FoldX-compatible residue sites.", df.head(150)


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


def save_pdf(summary_text, main_df, extra_df, fig):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_DIR / f"ABEVO_FoldX_pipeline_report_{ts}.pdf"

    with PdfPages(pdf_path) as pdf:
        fig1, ax1 = plt.subplots(figsize=(11, 8.5))
        ax1.axis("off")
        ax1.text(
            0.02,
            0.98,
            summary_text,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight")

        for title, df in [("Main Table", main_df), ("Extra Details", extra_df)]:
            if df is not None and not df.empty:
                fig2, ax2 = plt.subplots(figsize=(11, 8.5))
                ax2.axis("off")
                table = ax2.table(
                    cellText=df.head(20).values,
                    colLabels=list(df.columns),
                    loc="center",
                )
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
        raise ValueError("Please enter PDB chain ID, e.g., A, B, H, or L.")

    seq = read_fasta(fasta_path)
    if len(seq) > 1024:
        raise ValueError(
            f"Selected FASTA is too long: {len(seq)} residues. "
            "Please select a single-chain FASTA file."
        )

    cmd = [
        "python",
        str(RUNNERS_DIR / "run_abevo_foldx_pipeline.py"),
        str(fasta_path),
        str(pdb_path),
        chain_id,
        str(top_n),
    ]

    stdout, stderr, code = run_cmd(cmd)

    if code != 0:
        raise RuntimeError(stderr or stdout)

    result_dir = get_result_dir(fasta_path, pdb_path, chain_id)
    summary = json.loads((result_dir / "summary.json").read_text())

    ranked_csv = result_dir / "pipeline_ranked_results.csv"
    all_csv = result_dir / "pipeline_all_results.csv"

    ranked_df = pd.read_csv(ranked_csv) if ranked_csv.exists() else pd.DataFrame()
    all_df = pd.read_csv(all_csv) if all_csv.exists() else pd.DataFrame()

    if ranked_df.empty:
        main_df = all_df.head(20)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No valid FoldX-ranked mutations. Check mapping details.",
            ha="center",
        )
    else:
        main_df = ranked_df.head(30)
        fig = make_scatter(ranked_df)

    summary_text = (
        "ABEVO → FoldX Pipeline Report\n\n"
        f"FASTA: {fasta_path.name}\n"
        f"PDB: {pdb_path.name}\n"
        f"PDB chain ID: {chain_id}\n"
        f"Target number of FoldX-valid mutations: {top_n}\n"
        f"Successful FoldX results: {summary.get('successful_foldx_results')}\n"
        f"ABEVO cache: {summary.get('abevo_cache_status')}\n"
        f"Repair cache: {summary.get('repair_cache_status')}\n"
        f"FoldX batch status: {summary.get('foldx_batch_status')}\n\n"
        "Interpretation:\n"
        "1. Antibody-Evolution scores single-point mutations from the selected chain.\n"
        "2. Candidate mutations are mapped to the selected PDB chain.\n"
        "3. FoldX evaluates structurally valid mapped mutations.\n"
        "4. Final candidates are ranked using sequence/evolution score and FoldX energy."
    )

    pdf = save_pdf(summary_text, main_df, all_df.head(30), fig)

    return summary_text, main_df, fig, all_df.head(50), json.dumps(summary, indent=2), pdf


def parse_foldx_mutation(mutation):
    mutation = str(mutation).strip().replace(";", "")
    m = re.fullmatch(r"([A-Z])([A-Za-z0-9])(\d+)([A-Z])", mutation)
    if not m:
        raise ValueError("Mutation must look like NH73D, EB42G, RB44G, etc.")
    wt, chain, pos, mt = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    return wt, chain, pos, mt


def find_best_mutation_from_results(saved_fasta, saved_pdb, chain_id):
    fasta_path = BASE_DIR / saved_fasta
    pdb_path = BASE_DIR / saved_pdb
    result_dir = get_result_dir(fasta_path, pdb_path, chain_id.strip())

    ranked_csv = result_dir / "pipeline_ranked_results.csv"
    if not ranked_csv.exists():
        return ""

    df = pd.read_csv(ranked_csv)
    if df.empty or "mapped_foldx_mutation" not in df.columns:
        return ""

    return str(df.iloc[0]["mapped_foldx_mutation"])


def extract_structure_data(pdb_path, mutation_text):
    wt, mut_chain, mut_pos, mt = parse_foldx_mutation(mutation_text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_path))

    chain_coords = {}
    residue_points = []
    mut_coord = None
    mut_resname = None

    for model in structure:
        for chain in model:
            cid = str(chain.id)
            coords = []

            for residue in chain:
                if residue.id[0] != " ":
                    continue

                resseq = residue.id[1]
                resname = residue.get_resname().upper()

                if "CA" not in residue:
                    continue

                ca = residue["CA"].coord.astype(float)
                coords.append([ca[0], ca[1], ca[2], resseq, resname])

                residue_points.append({
                    "chain": cid,
                    "position": resseq,
                    "resname": resname,
                    "x": ca[0],
                    "y": ca[1],
                    "z": ca[2],
                })

                if cid == mut_chain and resseq == mut_pos:
                    mut_coord = ca
                    mut_resname = resname

            if coords:
                chain_coords[cid] = coords

        break

    if mut_coord is None:
        raise ValueError(
            f"Could not find mutation site {mutation_text}: chain {mut_chain}, residue {mut_pos}."
        )

    return {
        "wt": wt,
        "mutant": mt,
        "mut_chain": mut_chain,
        "mut_pos": mut_pos,
        "mut_coord": np.array(mut_coord, dtype=float),
        "mut_resname": mut_resname,
        "chain_coords": chain_coords,
        "residue_points": residue_points,
    }


def set_equal_axes_3d(ax, all_xyz):
    all_xyz = np.asarray(all_xyz)
    x_min, y_min, z_min = all_xyz.min(axis=0)
    x_max, y_max, z_max = all_xyz.max(axis=0)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


def make_overview_figure(pdb_path, mutation_text, data):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_xyz = []

    for cid, coords in data["chain_coords"].items():
        arr = np.array([[c[0], c[1], c[2]] for c in coords], dtype=float)
        all_xyz.append(arr)

        if cid == data["mut_chain"]:
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                linewidth=3.0,
                label=f"Selected chain {cid}",
            )
        else:
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                linewidth=1.0,
                alpha=0.30,
                label=f"Other chain {cid}",
            )

    mut = data["mut_coord"]
    ax.scatter(
        [mut[0]],
        [mut[1]],
        [mut[2]],
        s=300,
        marker="o",
        edgecolor="black",
        linewidth=1.5,
        label=f"Mutation site {mutation_text}",
    )

    ax.text(
        mut[0],
        mut[1],
        mut[2],
        f"{mutation_text}\n{data['mut_resname']} {data['mut_chain']}{data['mut_pos']}",
        fontsize=11,
        weight="bold",
    )

    all_xyz = np.vstack(all_xyz)
    set_equal_axes_3d(ax, all_xyz)

    ax.set_title(f"Whole-structure overview: {pdb_path.name}\nHighlighted mutation: {mutation_text}", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-55)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    return fig


def make_zoom_figure(pdb_path, mutation_text, data, radius=12.0):
    mut = data["mut_coord"]

    nearby = []
    for r in data["residue_points"]:
        coord = np.array([r["x"], r["y"], r["z"]], dtype=float)
        dist = np.linalg.norm(coord - mut)
        if dist <= radius:
            nearby.append({**r, "distance": dist})

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if not nearby:
        ax.text2D(0.5, 0.5, "No nearby residues found.", ha="center")
        return fig

    for r in nearby:
        coord = np.array([r["x"], r["y"], r["z"]], dtype=float)
        if r["chain"] == data["mut_chain"] and r["position"] == data["mut_pos"]:
            ax.scatter(coord[0], coord[1], coord[2], s=350, marker="*", edgecolor="black", linewidth=1.5)
            ax.text(coord[0], coord[1], coord[2], mutation_text, fontsize=12, weight="bold")
        else:
            ax.scatter(coord[0], coord[1], coord[2], s=80, alpha=0.75)
            if r["distance"] <= 7.0:
                aa1 = AA3_TO_1.get(r["resname"], r["resname"])
                ax.text(coord[0], coord[1], coord[2], f"{aa1}{r['chain']}{r['position']}", fontsize=8)

    nearby_xyz = np.array([[r["x"], r["y"], r["z"]] for r in nearby], dtype=float)
    set_equal_axes_3d(ax, nearby_xyz)

    ax.set_title(
        f"Zoomed mutation neighborhood within {radius:.0f} Å\n{mutation_text} in {pdb_path.name}",
        fontsize=14,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=-45)
    plt.tight_layout()
    return fig


def make_distance_figure(pdb_path, mutation_text, data, radius=15.0):
    mut = data["mut_coord"]

    rows = []
    for r in data["residue_points"]:
        coord = np.array([r["x"], r["y"], r["z"]], dtype=float)
        dist = float(np.linalg.norm(coord - mut))

        if dist <= radius and not (r["chain"] == data["mut_chain"] and r["position"] == data["mut_pos"]):
            aa1 = AA3_TO_1.get(r["resname"], r["resname"])
            rows.append({
                "label": f"{aa1}{r['chain']}{r['position']}",
                "distance": dist,
                "chain": r["chain"],
                "position": r["position"],
            })

    rows = sorted(rows, key=lambda x: x["distance"])[:20]

    fig, ax = plt.subplots(figsize=(10, 5))

    if not rows:
        ax.text(0.5, 0.5, "No neighboring residues found.", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [r["label"] for r in rows]
    distances = [r["distance"] for r in rows]

    ax.bar(labels, distances)
    ax.axhline(5.0, linestyle="--", linewidth=1.2, label="~5 Å close-contact guide")
    ax.set_title(f"Nearest residues around mutation {mutation_text}")
    ax.set_xlabel("Nearby residue")
    ax.set_ylabel("Distance from mutation CA atom (Å)")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def save_structure_figures(pdb_path, mutation_text, overview_fig, zoom_fig, distance_fig):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    overview_path = STRUCTURE_DIR / f"{safe_name(pdb_path.stem)}_{safe_name(mutation_text)}_overview_{ts}.png"
    zoom_path = STRUCTURE_DIR / f"{safe_name(pdb_path.stem)}_{safe_name(mutation_text)}_zoom_{ts}.png"
    distance_path = STRUCTURE_DIR / f"{safe_name(pdb_path.stem)}_{safe_name(mutation_text)}_distances_{ts}.png"

    overview_fig.savefig(overview_path, dpi=300, bbox_inches="tight")
    zoom_fig.savefig(zoom_path, dpi=300, bbox_inches="tight")
    distance_fig.savefig(distance_path, dpi=300, bbox_inches="tight")

    return str(overview_path), str(zoom_path), str(distance_path)


def generate_structure_figures(saved_pdb, mutation_text, chain_id, saved_fasta):
    pdb_path = BASE_DIR / saved_pdb

    if not mutation_text or not mutation_text.strip():
        mutation_text = find_best_mutation_from_results(saved_fasta, saved_pdb, chain_id)

    if not mutation_text:
        raise ValueError(
            "No mutation was provided and no ranked result was found. "
            "Run the pipeline first or enter a mutation such as NH73D."
        )

    data = extract_structure_data(pdb_path, mutation_text)

    overview_fig = make_overview_figure(pdb_path, mutation_text, data)
    zoom_fig = make_zoom_figure(pdb_path, mutation_text, data, radius=12.0)
    distance_fig = make_distance_figure(pdb_path, mutation_text, data, radius=15.0)

    overview_path, zoom_path, distance_path = save_structure_figures(
        pdb_path,
        mutation_text,
        overview_fig,
        zoom_fig,
        distance_fig,
    )

    wt, mut_chain, mut_pos, mt = parse_foldx_mutation(mutation_text)

    info = (
        f"Generated three structure figures\n"
        f"PDB: {pdb_path}\n"
        f"Mutation: {mutation_text}\n"
        f"Highlighted site: chain {mut_chain}, position {mut_pos}, WT {wt}, mutant {mt}\n"
        f"Overview PNG: {overview_path}\n"
        f"Zoom PNG: {zoom_path}\n"
        f"Distance PNG: {distance_path}"
    )

    return overview_fig, zoom_fig, distance_fig, overview_path, zoom_path, distance_path, info


def find_best_mutation_from_results(saved_fasta, saved_pdb, chain_id):
    fasta_path = BASE_DIR / saved_fasta
    pdb_path = BASE_DIR / saved_pdb
    result_dir = get_result_dir(fasta_path, pdb_path, chain_id.strip())

    ranked_csv = result_dir / "pipeline_ranked_results.csv"
    if not ranked_csv.exists():
        return ""

    df = pd.read_csv(ranked_csv)
    if df.empty or "mapped_foldx_mutation" not in df.columns:
        return ""

    return str(df.iloc[0]["mapped_foldx_mutation"])


fastas = list_fastas()
pdbs = list_pdbs()

with gr.Blocks(title="ABEVO → FoldX Antibody Mutation Prioritization App") as demo:
    gr.Markdown("# ABEVO → FoldX Antibody Mutation Prioritization App")
    gr.Markdown(
        "This focused app runs everything from the frontend: "
        "ABEVO mutation scoring → FoldX structural filtering → ranked report → structure figures."
    )

    with gr.Row():
        with gr.Column(scale=1):
            saved_fasta = gr.Dropdown(fastas, value=fastas[0], label="Choose FASTA chain")
            saved_pdb = gr.Dropdown(pdbs, value=pdbs[0], label="Choose matching PDB")
            chain_id = gr.Textbox(label="PDB Chain ID", placeholder="Example: H, L, A, or B", value="H")

            top_n = gr.Number(
                label="Target number of FoldX-valid mutations",
                value=5,
                precision=0,
            )

            scan_btn = gr.Button("Scan Valid FoldX Sites")
            run_btn = gr.Button("Run Full ABEVO → FoldX Pipeline")

            gr.Markdown("### Structure figures")
            mutation_for_fig = gr.Textbox(
                label="Mutation to visualize",
                placeholder="Leave blank to use top-ranked mutation, or enter NH73D",
            )
            fig_btn = gr.Button("Generate Structure Figures")

            report_file = gr.File(label="Download PDF Report")

        with gr.Column(scale=2):
            with gr.Tab("Important Summary"):
                summary = gr.Textbox(label="Key Findings", lines=15)

            with gr.Tab("Ranked Candidates"):
                main_table = gr.Dataframe(label="Final Ranked Mutations")

            with gr.Tab("Main Plot"):
                plot = gr.Plot(label="ABEVO vs FoldX Plot")

            with gr.Tab("Mapping Details"):
                extra = gr.Dataframe(label="All Mapping/FoldX Details")

            with gr.Tab("Structure Overview"):
                overview_plot = gr.Plot(label="Whole Structure Overview")
                overview_file = gr.File(label="Download Overview PNG")

            with gr.Tab("Mutation Zoom"):
                zoom_plot = gr.Plot(label="Zoomed Mutation Neighborhood")
                zoom_file = gr.File(label="Download Zoom PNG")

            with gr.Tab("Nearby Distances"):
                distance_plot = gr.Plot(label="Nearest Residues Distance Plot")
                distance_file = gr.File(label="Download Distance PNG")

            with gr.Tab("Structure Info"):
                structure_info = gr.Textbox(label="Structure Figure Info", lines=8)

            with gr.Tab("Valid FoldX Sites"):
                scan_summary = gr.Textbox(label="Scan Summary", lines=3)
                scan_table = gr.Dataframe(label="Available PDB Residue Sites")

            with gr.Tab("Raw JSON"):
                raw_json = gr.Code(label="Summary JSON", language="json")

    scan_btn.click(
        scan_pdb_sites,
        inputs=[saved_pdb],
        outputs=[scan_summary, scan_table],
    )

    run_btn.click(
        run_pipeline,
        inputs=[saved_fasta, saved_pdb, chain_id, top_n],
        outputs=[summary, main_table, plot, extra, raw_json, report_file],
    )

    fig_btn.click(
        generate_structure_figures,
        inputs=[saved_pdb, mutation_for_fig, chain_id, saved_fasta],
        outputs=[
            overview_plot,
            zoom_plot,
            distance_plot,
            overview_file,
            zoom_file,
            distance_file,
            structure_info,
        ],
    )

demo.launch(
    allowed_paths=[
        str(BASE_DIR),
        str(RESULTS_DIR),
        str(REPORT_DIR),
        str(STRUCTURE_DIR),
    ]
)
