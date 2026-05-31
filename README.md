AI-assisted antibody engineering platform combining Antibody Evolution (ABEVO), FoldX structural analysis, interface-aware mutation prioritization, protein structure visualization, and interactive Gradio workflows.

# ABEVO-FoldX-AI-Antibody-Design

Interactive AI-assisted antibody engineering platform combining sequence-evolution scoring, structural stability analysis, interface-aware filtering, and mutation prioritization.

---

## Overview

This project integrates:

- Antibody Evolution (ABEVO) mutation scoring
- FoldX structural energy evaluation
- Persistent caching for rapid re-analysis
- Antibody-antigen interface detection
- Optional CDR-focused mutation restriction
- Protein structure visualization
- Interactive Gradio frontend

The goal is to prioritize antibody mutations that are promising from both sequence-evolution and structural perspectives.

---

## Features

### Sequence-Based Mutation Scoring

Generate mutation candidates using Antibody Evolution (ABEVO).

- Scores all possible single-point mutations
- Identifies evolutionarily favorable substitutions
- Produces ranked mutation lists

---

### FoldX Structural Filtering

Evaluate candidate mutations structurally.

- Mutation mapping from sequence to PDB
- Structural stability estimation
- FoldX energy-based filtering
- Candidate prioritization

---

### Persistent Caching

Repeated analyses become significantly faster.

Caches:

- ABEVO mutation predictions
- Repaired FoldX structures
- FoldX mutation results

Results remain available across sessions and machine restarts.

---

### Interface-Aware Antibody Design

Detect residues located near antibody-antigen interfaces.

Capabilities:

- Interface residue identification
- Distance calculations
- Interface-only mutation prioritization
- Optional interface filtering

---

### Optional CDR Restriction

Restrict mutation search to:

- User-defined CDR regions
- Interface residues
- Interface + CDR intersection

---

### Protein Structure Visualization

Generate structural figures showing:

- Antibody chains
- Selected mutation sites
- Mutation location within the structure

Useful for interpretation and presentation.

---

## Workflow

```text
FASTA Sequence
      │
      ▼
Antibody Evolution (ABEVO)
      │
      ▼
Mutation Ranking
      │
      ▼
Sequence → Structure Mapping
      │
      ▼
FoldX Evaluation
      │
      ▼
Candidate Filtering
      │
      ▼
Interface/CDR Filtering (Optional)
      │
      ▼
Final Ranked Mutations
      │
      ▼
Structure Visualization
```

---

## Applications

### Antibody Optimization

Prioritize mutations that may improve:

- Stability
- Manufacturability
- Structural robustness

---

### Interface Analysis

Identify mutations near:

- Antibody-antigen contacts
- Potential binding regions

---

### Candidate Selection

Reduce thousands of candidate mutations into a small list suitable for experimental validation.

---

## Project Structure

```text
project/
│
├── app/
│   ├── app_abevo_foldx.py
│   ├── app_abevo_foldx_structure.py
│   └── app_abevo_foldx_interface.py
│
├── runners/
│   ├── run_abevo_foldx_pipeline.py
│   └── run_interface_pipeline.py
│
├── data/
│   ├── fasta/
│   └── pdb/
│
├── results/
│
├── cache/
│
└── reports/
```

---

## Installation

### Create Environment

```bash
conda create -n antibody_design python=3.10
conda activate antibody_design
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Main Application

```bash
cd app

python app_abevo_foldx.py
```

Open:

```text
http://127.0.0.1:7860
```

---

## Running the Structure Visualization App

```bash
python app_abevo_foldx_structure.py
```

---

## Running the Interface-Aware App

```bash
python app_abevo_foldx_interface.py
```

---

## Example Output

### Ranked Mutation Candidates

| Mutation | FoldX Mutation |
|-----------|---------------|
| I42A | IA43A |
| I42S | IA43S |
| F134L | FA135L |

---

### Interface Residues

| Chain | Position | Distance |
|---------|-----------|------------|
| A | 36 | 1.89 Å |
| A | 38 | 1.82 Å |
| A | 43 | 3.56 Å |

---

## Screenshots

### Main Dashboard

![Dashboard](images/dashboard.png)

### Ranked Mutation Candidates

![Ranked Candidates](images/ranked_candidates.png)

### Structure Visualization

![Structure](images/structure_visualization.png)

### Interface Residues

![Interface](images/interface_residues.png)

### ABEVO vs FoldX Plot

![Plot](images/mutation_plot.png)

---

## Current Limitations

- FoldX predictions are computational estimates
- Experimental validation is required
- Interface analysis requires true antibody-antigen complexes
- No de novo antibody generation yet
- No binding-affinity optimization yet

---

## Future Directions

Planned extensions:

- Antibody-antigen binding energy optimization
- Automated CDR identification
- De novo antibody generation
- Generative protein language models
- Active-learning guided design
- Experimental feedback integration

---

## Disclaimer

This software is intended for research and educational purposes only.

Predicted mutations should not be interpreted as experimentally validated improvements without laboratory testing.

---

## Author

Shyam Sundar Debsarkar

PhD Candidate, Computer Science

University of Cincinnati

Research Areas:

- AI for Biomedical Informatics
- Computational Biology
- Protein Engineering
- Antibody Design
- Machine Learning
- Deep Learning
