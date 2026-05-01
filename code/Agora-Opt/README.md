# Agora-Opt Code Package

## What This Directory Contains

`./code/Agora-Opt/` is the source directory for the Agora-Opt method. It
retains two categories of assets:

- the Agora-Opt implementation
- prebuilt memory assets used by the method

Historical run outputs are not stored here.

For compatibility with the original stage naming, the main reproduction script
maintains two convenience paths:

- `generated_with_memory`
- `debate_runs`

## Important Subdirectories

The most important components are:

- `src/debate_memory/`: core Agora-Opt implementation
- `scripts/`: command-line wrappers
- `memory_storage/`: solution memory
- `debug_case_memory/`: debug memory retrieval bank
- `debate_memory_storage/`: debate memory retrieval bank
- `memory_variants/`: retained alternative memory variants
- `memory_backups/`: retained memory backups

Multiple memory versions are intentionally kept. They were prepared during
different stages of the project and can all be treated as available assets for
generation, debugging, and debate.

## Core Workflow

Agora-Opt runs in two stages.

### Stage 1: Generate Initial Solutions

`generate_with_memory.py` generates candidate solutions, optionally using
solution memory and debug memory.

Primary entry script:

- `scripts/generate_with_memory.py`

This stage:

- reads benchmark problems
- retrieves similar solved cases from `memory_storage/`
- generates candidate modeling code
- uses debug memory during self-repair when execution fails

### Stage 2: Run Debate

`run_memory_debate.py` takes two sets of initial solutions and runs the
decentralized debate stage.

Primary entry script:

- `scripts/run_memory_debate.py`

This stage:

- loads both sides' initial solutions
- retrieves historical debate cases from `debate_memory_storage/`
- performs iterative comparison, revision, and convergence
- executes and evaluates the final consensus solution

## Memory Types

### 1. Solution Memory

Directory:

- `memory_storage/`

Purpose:

- retrieves similar successful modeling cases during generation
- supplies formulation templates and structural priors

Build path:

- extract `(problem description, correct code, objective value)` from correctly
  evaluated runs
- build `cases.jsonl` plus its retrieval index

Related script:

- `scripts/build_memory_from_eval_results.py`

### 2. Debug Memory

Directory:

- `debug_case_memory/`

Purpose:

- retrieves similar execution failures and repair experience
- supports automatic self-debugging during generation

Build path:

- extract unique error signatures from `debug_memory.jsonl` and its backups
- normalize the error text, repair hints, and metadata into a retrieval bank

Related script:

- `scripts/build_debug_memory.py`

Note:

- raw debug logs are stored in `memory_storage/debug_memory.jsonl`
- that log file is one of the inputs used to build debug memory

### 3. Debate Memory

Directory:

- `debate_memory_storage/`

Purpose:

- stores examples of how disagreements were resolved during debate
- helps later debates converge more efficiently

Build path:

- select historical runs where the two initial solutions disagreed
- keep cases where debate eventually converged successfully
- extract the dispute, key arguments, and final converged code

Related scripts:

- `scripts/build_debate_memory.py`
- `scripts/process_all_debate_cases.sh`

## Suggested Memory Construction Order

When preparing memory from scratch, the recommended order is:

1. run generation and evaluation to obtain `evaluation_results`
2. build solution memory from correct cases
3. build debug memory from accumulated `debug_memory.jsonl`
4. build debate memory from historical debate runs

The dependency flow is:

- `evaluation_results` -> `solution memory`
- `debug_memory.jsonl` -> `debug memory`
- debate run artifacts -> `debate memory`

## Retained Memory Assets

This directory intentionally keeps:

- the three primary memory stores
- memory variants
- memory backups

These are treated as static method assets.

Historical run outputs are not retained here, which keeps source code, memory
assets, and new results clearly separated.

To rebuild the three memory types, use:

```bash
bash ./code/Agora-Opt/scripts/build_memory_assets.sh /path/to/eval_dir1 /path/to/eval_dir2
```

That script attempts to:

- rebuild solution memory from evaluation directories
- rebuild debug memory from `debug_memory.jsonl` and its backups
- rebuild debate memory from debate run artifacts

## Recommended Entry Points

For paper reproduction, use the outer scripts rather than manually assembling
commands in this directory:

- main table: `./code/scripts/run_agora.sh`
- 5.1: `./code/experiments/5.1_compatibility_backbone_llms/`
- 5.2: `./code/experiments/5.2_ablation_study/`
- 5.3.1: `./code/experiments/5.3.1_centralized_judge_selection/`
- 5.3.2: `./code/experiments/5.3.2_impact_of_debate_rounds/`
- 5.3.3:
  `./code/experiments/5.3.3_generalization_of_decentralized_debate_protocol/`

## Direct Source-Level Usage

For direct method-level use, the main wrappers are:

```bash
python scripts/generate_with_memory.py
python scripts/run_memory_debate.py
python scripts/execute.py
python scripts/build_memory_from_eval_results.py
python scripts/build_debug_memory.py
python scripts/build_debate_memory.py
```

## Path Conventions

Within the open-source package, the intended layout is:

- benchmark data: `./data/benchmarks/`
- Agora-Opt source code and memory: `./code/Agora-Opt/`

This separation makes the boundaries between code, memory assets, and newly
generated outputs explicit.
