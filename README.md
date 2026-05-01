# Agora-Opt: Open Data and Code Package

## Overview

This package contains the data, code, and experiment scripts used for the
Agora-Opt project. It is organized for two common use cases:

- understanding which benchmarks, methods, and scripts correspond to the paper
- reproducing the reported results under a unified project structure

The package includes:

- standardized benchmark datasets
- runnable code for Agora-Opt and several baselines
- experiment entry scripts for the main results and Section 5 analyses

For a first pass through the repository, the recommended reading order is:

1. `Directory Layout`
2. `What Is Included`
3. `Reproducing Results`

## Quick Start

To run a small end-to-end Agora-Opt example without relying on prebuilt memory,
set the API environment variables first:

```bash
export LLM_API_KEY="PUT YOUR API KEY HERE"
export LLM_API_BASE_URL="PUT YOUR API URL HERE"
```

Then run:

```bash
bash ./code/scripts/start.sh
```

This script:

- runs a small example on `EasyLP` by default
- skips solution memory, debug memory, and debate memory
- executes both generation and debate

An explicit dataset and sample count can also be provided:

```bash
bash ./code/scripts/start.sh ComplexLP 10
```

To reproduce the main Agora-Opt table instead of a lightweight smoke test, run:

```bash
bash ./code/scripts/run_agora.sh
```

That script uses the retained Agora-Opt memory assets.

For Section 5 analysis scripts, see:

- `./code/experiments/5.1_compatibility_backbone_llms/`
- `./code/experiments/5.2_ablation_study/`
- `./code/experiments/5.3.1_centralized_judge_selection/`
- `./code/experiments/5.3.2_impact_of_debate_rounds/`
- `./code/experiments/5.3.3_generalization_of_decentralized_debate_protocol/`

Detailed prerequisites for each experiment are documented in:

- `./code/scripts/experiment_guide.md`

## What Is Included

### Benchmarks

Standardized benchmark files are located in `./data/benchmarks/`:

- `NL4OPT.jsonl`
- `EasyLP.jsonl`
- `ComplexLP.jsonl`
- `NLP4LP.jsonl`
- `ComplexOR.jsonl`
- `IndustryOR.jsonl`
- `ReSocratic.jsonl`
- `OPT-Principled.jsonl`

Within this package:

- `IndustryOR` is the normalized name for the earlier `IndustryOR_fixedV2_clean`
- `OPT-Principled` is the normalized name for the earlier
  `combined` / `combined_dataset`

The open-source package consistently uses the normalized names above and no
longer relies on the `_clean` suffix.

### Method Code

Agora-Opt source code is located in `./code/Agora-Opt/`.
Other baselines are located in `./code/baseline/`.

### Experiment Scripts

Unified run scripts are located in:

- `./code/scripts/`

Section 5 experiment scripts are located in:

- `./code/experiments/`

### Memory Assets

Agora-Opt depends on three memory types, all of which are retained under
`./code/Agora-Opt/`:

- `memory_storage/`: solution memory
- `debug_case_memory/`: debug memory retrieval bank
- `debate_memory_storage/`: debate memory retrieval bank

The following directories are also retained as supporting assets:

- `memory_variants/`
- `memory_backups/`

Historical run directories are not kept inside the Agora-Opt source tree.

Multiple retained memory versions are included. They originate from different
experiment preparation stages and can be treated as available assets for the
generation, debugging, and debate stages.

For compatibility with older stage semantics, `run_agora.sh` also maintains:

- `./code/Agora-Opt/generated_with_memory`
- `./code/Agora-Opt/debate_runs`

## Directory Layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── benchmarks/
├── code/
│   ├── Agora-Opt/
│   ├── baseline/
│   ├── scripts/
│   └── experiments/
└── models/
```

The intended interpretation is:

- `data/benchmarks/`: benchmark data used by the paper
- `code/Agora-Opt/`: Agora-Opt source code and memory assets
- `code/baseline/`: baseline implementations
- `code/scripts/`: main table and baseline entry scripts
- `code/experiments/`: scripts for Sections 5.1, 5.2, and 5.3
- `models/`: recommended location for local third-party model assets

## Environment Setup

### Python and Dependencies

Python 3.10+ is recommended.

The repository dependency snapshot is exported at:

- `./requirements.txt`

Core dependencies include:

- `gurobipy`
- `openai` or an API-compatible client
- `numpy`
- `pandas`
- `tqdm`
- `sentence-transformers`
- vector retrieval dependencies used by Agora-Opt memory

Some local-judge or training-centric workflows may also require:

- `vllm`
- CUDA-related dependencies

### API Configuration

All scripts that call remote LLM services should read credentials from
environment variables:

```bash
export LLM_API_KEY="PUT YOUR API KEY HERE"
export LLM_API_BASE_URL="PUT YOUR API URL HERE"
```

For compatibility across different code paths, some scripts also read:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `API_KEY`
- `API_URL`

## Reproducing Results

### Main Agora-Opt Table

To reproduce the main Agora-Opt results over all eight benchmarks:

```bash
bash ./code/scripts/run_agora.sh
```

The script explicitly chains two stages:

1. generate `initial solution` files for the two backbones
2. run `run_memory_debate.py` for debate and final evaluation

Outputs are written to the current working directory or a path you specify.

### Section 5 Experiments

| Paper section | Experiment | Entry script |
| --- | --- | --- |
| 5.1 | Compatibility w.r.t. Backbone LLMs | `./code/experiments/5.1_compatibility_backbone_llms/run_agora_backbone_compatibility.sh` |
| 5.2 | Ablation Study | `./code/experiments/5.2_ablation_study/run_agora_ablation.sh` |
| 5.3.1 | Decentralized Debate vs. Centralized Judge | `./code/experiments/5.3.1_centralized_judge_selection/run_centralized_judge_selection.sh` |
| 5.3.2 | Impact of Debate Rounds | `./code/experiments/5.3.2_impact_of_debate_rounds/run_debate_rounds.sh` |
| 5.3.3 | Generalization of Decentralized Debate Protocol | `./code/experiments/5.3.3_generalization_of_decentralized_debate_protocol/run_decentralized_debate_generalization.sh` |

Before running any Section 5 script, read:

- `./code/scripts/experiment_guide.md`

That guide explains:

- whether the script already runs both Agora stages internally
- whether candidate `initial solution` files must be prepared in advance
- whether external memory paths must be supplied manually

### Baseline Entry Scripts

Standalone baseline entry points include:

- `./code/scripts/run_zero_shot.sh`
- `./code/scripts/run_cot.sh`
- `./code/scripts/run_coe.sh`
- `./code/scripts/run_cafa.sh`
- `./code/scripts/run_optimus.sh`

These scripts read benchmark inputs from `./data/benchmarks/`.

## Paper-to-Code Map

For a direct "where should I look for this result?" mapping:

### Overall Performance

- method implementation: `./code/Agora-Opt/`
- main table entry script: `./code/scripts/run_agora.sh`

### 5.1 Compatibility w.r.t. Backbone LLMs

- directory: `./code/experiments/5.1_compatibility_backbone_llms/`
- purpose: swaps backbone combinations while reusing the full Agora-Opt pipeline

### 5.2 Ablation Study

- directory: `./code/experiments/5.2_ablation_study/`
- purpose: removes debate memory, debug memory, solution memory, the debate
  mechanism, or the agent team one at a time

### 5.3.1 Decentralized Debate vs. Centralized Judge

- directory: `./code/experiments/5.3.1_centralized_judge_selection/`
- purpose: compares decentralized debate against explicit judge selection
- note: requires candidate `initial solution` files and their evaluation files

### 5.3.2 Impact of Debate Rounds

- directory: `./code/experiments/5.3.2_impact_of_debate_rounds/`
- purpose: keeps the same `initial solution` files while changing only
  `max_rounds`

### 5.3.3 Generalization of Decentralized Debate Protocol

- directory:
  `./code/experiments/5.3.3_generalization_of_decentralized_debate_protocol/`
- purpose: reuses only the Agora debate stage on top of external methods'
  `initial solution` files

## Notes on Memory

Agora-Opt uses three memory types:

- `solution memory`: retrieves similar formulation cases during generation
- `debug memory`: provides historical debugging experience after execution
  failures
- `debate memory`: provides historical disagreement-resolution experience during
  debate

This package retains the built memory assets and selected variant / backup
directories so the project can be reproduced or extended without rebuilding
everything from scratch.

The retained assets include multiple versions and can be understood as different
prepared instances of solution memory, debug memory, and debate memory.

For Agora-specific details, see:

- `./code/Agora-Opt/README.md`

## Assets Not Distributed Here

Users must provide the following separately:

- commercial model API keys
- private or restricted API endpoints
- StepORLM / GenPRM weights
- any third-party model assets restricted by license

`./models/` is the recommended location for local third-party model assets,
including StepORLM / GenPRM or other locally hosted models used by selected
experiments. This package does not redistribute those repositories or weights.

## Reproducibility Notes

- API-backed model versions may change over time, so exact historical numbers
  may shift slightly.
- Multi-threading, backend changes, and temperature settings can introduce
  variance.
- Some experiment scripts require users to fill in `initial solution` paths or
  local model paths before running.
- For formal reproduction, record the model name, provider, run date, and key
  parameters.
