# Local Model Assets

`./models/` is the recommended location for local third-party model assets
used by selected experiments in this package.

Examples include:

- StepORLM / GenPRM assets for `5.3.1_centralized_judge_selection`
- other locally served judge models
- additional research models required by external-method experiments

This package does not redistribute those repositories, checkpoints, or weights.

Recommended usage:

1. Place the required local model assets somewhere under `./models/`.
2. Point the corresponding experiment script to the exact subdirectory or model
   path through its environment variables or command-line arguments.
3. Launch the required local serving stack, such as `vllm`, from that path when
   needed.
