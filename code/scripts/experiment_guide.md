# Experiment Reproduction Guide

This document explains how each experiment script relates to the required
prerequisites, with two main questions in mind:

- what must be prepared before running the script
- where Stage 1 `initial solution` outputs connect to Stage 2 `debate`

## Agora-Opt Main Pipeline

Main entry point:

- `./code/scripts/run_agora.sh`

This script already chains the two Agora stages end to end:

1. Stage 1: `generate_with_memory.py`
2. Stage 2: `run_memory_debate.py`

The handoff is explicit:

- Stage 1 output: `./results/Agora-Opt/generation/`
- Stage 2 input: the same directory, passed directly into debate

To preserve the earlier directory semantics, the script also maintains:

- `./code/Agora-Opt/generated_with_memory` ->
  `./results/Agora-Opt/generation`
- `./code/Agora-Opt/debate_runs` -> `./results/Agora-Opt/debate`

As a result, the Agora-Opt main pipeline is the most complete and direct
reproduction path in this package.

## 5.1 Compatibility w.r.t. Backbone LLMs

Entry point:

- `./code/experiments/5.1_compatibility_backbone_llms/run_agora_backbone_compatibility.sh`

Prerequisites:

- none beyond API configuration

Notes:

- reuses `run_agora.sh`
- runs three backbone combinations
- uses the full two-stage Agora-Opt pipeline
- uses the retained Agora-Opt memory assets

Datasets:

- `NL4OPT`
- `EasyLP`
- `ComplexLP`
- `NLP4LP`
- `ComplexOR`
- `IndustryOR`
- `ReSocratic`
- `OPT-Principled`

## 5.2 Ablation Study

Entry point:

- `./code/experiments/5.2_ablation_study/run_agora_ablation.sh`

Prerequisites:

- none beyond API configuration

Notes:

- the script runs both generation and debate / evaluation internally
- each variant is implemented by switching off a specific memory source or
  mechanism

Datasets:

- `ComplexLP`
- `IndustryOR`
- `OPT-Principled`

Variant interpretation:

- `full_agora_opt`: full Agora-Opt
- `no_debate_memory`: no debate memory during the debate stage
- `no_debug_memory`: no debug-memory retrieval during generation
- `no_solution_memory`: no solution-memory retrieval during generation
- `no_debate_single_agent`: generation and evaluation only, no debate
- `vanilla_gpt4o`: direct zero-shot GPT-4o baseline

## 5.3.1 Decentralized Debate vs. Centralized Judge

Entry point:

- `./code/experiments/5.3.1_centralized_judge_selection/run_centralized_judge_selection.sh`

Prerequisites:

1. prepare candidate `initial solution` files for both sides
2. run `execute.py` separately on both candidate sets to obtain
   `evaluation_results.jsonl`
3. fill those paths into the shell script placeholders

Notes:

- this experiment does not generate candidate solutions automatically
- it expects explicit inputs for:
  - candidate `initial solution` JSONL files
  - their corresponding `evaluation_results.jsonl`
- centralized judge or GenPRM-style selection is then applied on top

Recommended preparation:

- use the Agora generation stage to prepare GPT-4o Team and DeepSeek-V3 Team
  candidates
- evaluate them separately with `execute.py`

Datasets:

- `ComplexLP`
- `IndustryOR`
- `OPT-Principled`

Memory usage:

- candidate generation should follow the standard Agora generation pipeline
- the judge stage itself does not rerun the full Agora debate procedure

## 5.3.2 Impact of Debate Rounds

Entry point:

- `./code/experiments/5.3.2_impact_of_debate_rounds/run_debate_rounds.sh`

Prerequisites:

- none beyond API configuration

Notes:

- the script first generates one set of two-sided `initial solution` files
- it then reuses those same candidates while varying `max_rounds=0,1,2,3`
- Stage 1 and Stage 2 are already connected correctly inside the script

Datasets:

- `NL4OPT`
- `EasyLP`
- `ComplexLP`
- `IndustryOR`
- `OPT-Principled`

Memory usage:

- uses the retained Agora-Opt solution, debug, and debate memory assets

## 5.3.3 Generalization of Decentralized Debate Protocol

Entry point:

- `./code/experiments/5.3.3_generalization_of_decentralized_debate_protocol/run_decentralized_debate_generalization.sh`

Prerequisites:

1. prepare Stage 1 `initial solution` files for the four external methods
2. prepare the memory paths to be supplied to the Agora debate stage
3. fill those paths into the environment-variable placeholders in the script

Notes:

- this experiment reuses only the Agora debate protocol
- it does not generate Stage 1 outputs for OptiMUS, CAFA, ORLM, or StepORLM
- the script reorganizes the provided `initial solution` files into the format
  expected by `run_memory_debate.py`, then runs debate

External methods:

- `OptiMUS-v0.3`
- `CAFA`
- `ORLM`
- `StepORLM`

Datasets:

- `NL4OPT`
- `EasyLP`
- `ComplexLP`
- `NLP4LP`
- `ComplexOR`
- `IndustryOR`
- `ReSocratic`
- `OPT-Principled`

Memory usage:

- requires explicit solution-memory, debug-memory, and debate-memory paths
- multiple retained memory versions are available in the package as usable
  assets for these stages

## Quick Start vs. Main Reproduction

- `./code/scripts/start.sh`: no memory, small-scale, intended to verify that
  the pipeline runs
- `./code/scripts/run_agora.sh`: full Agora-Opt main-table reproduction

If the goal is to verify that the code path works end to end, start with
`start.sh`.

If the goal is to reproduce the main paper result, use `run_agora.sh`.
