"""
Configuration file for simple RAG evaluation
Contains prompt templates and other settings
"""

from pathlib import Path

# ============================================
# Prompt Templates
# ============================================

# Default Gurobi prompt template
GUROBI_PROMPT = {
    "system": """You are a helpful Assistant with expertise in mathematical modeling and the Gurobi solver. When the User provides an OR question, you will analyze it, build a detailed mathematical model, and provide the Gurobi code to solve it.

Your response should follow these steps:
1. Carefully analyze the problem to identify decision variables, objective, and constraints.

2. Develop a complete mathematical model, explicitly defining:
    - Sets
    - Parameters
    - Decision Variables (and their types)
    - Objective Function
    - Constraints
3. Provide the corresponding Gurobi Python code to implement the model.

Implementation guardrails:
- Use `gurobipy` exclusively (avoid cvxpy/pulp/copty imports).
- When indexing tupledict variables across periods, introduce an explicit sentinel index (e.g., period 0) for initial conditions instead of accessing undefined keys like `x[-1]`.
- Define any Big-M constants explicitly using bounds derived from the data before they appear in constraints.
- Keep the model linear/integer; if a relationship seems non-linear, introduce auxiliary variables and linearization rather than exponentiation or log constraints.
- Always ensure every symbol referenced in constraints/objective (such as `M`, helper dictionaries, etc.) is declared in the code block.
""",
    "user": """Problem:
{question}

Provide a complete solution with mathematical model and Gurobi code.
"""
}

# ============================================
# Model Configuration
# ============================================

# Supported models and their default temperatures
MODEL_CONFIGS = {
    "gpt-4o": {"temperature": 0.01, "max_tokens": 8192},
    "gpt-4o-mini": {"temperature": 0.01, "max_tokens": 8192},
    "deepseek-chat": {"temperature": 0.01, "max_tokens": 8192},
    "gemini-2.0-flash-exp": {"temperature": 0.01, "max_tokens": 8192},
    "gemini-2.5-pro": {"temperature": 0.01, "max_tokens": 8192},
}

# ============================================
# Evaluation Configuration
# ============================================

EVAL_CONFIG = {
    # Execution settings
    "timeout": 60,  # seconds
    "max_retries": 3,
    
    # Answer comparison settings
    "tolerance": 0.05,  # 5% relative tolerance by default
    "use_relative_tolerance": True,
    "absolute_tolerance": 1e-3,  # for zero objective values
    
    # Output settings
    "save_code": True,
    "save_output": False,  # whether to save stdout/stderr
    "verbose": False,
}

# ============================================
# Dataset Configuration
# ============================================

# Supported datasets
DATASETS = [
    "ComplexLP",
    "EasyLP",
    "IndustryOR",
    "NL4OPT",
    "NLP4LP",
    "ReSocratic",
    "ComplexOR",
    "OPT-Principled",
]

DATASET_ALIASES = {
    "complexlp_clean": "ComplexLP",
    "easylp_clean": "EasyLP",
    "industryor_clean": "IndustryOR",
    "industryor_v2": "IndustryOR",
    "industryor_fixedv2": "IndustryOR",
    "industryor_fixedv2_clean": "IndustryOR",
    "nl4opt": "NL4OPT",
    "nl4opt_clean": "NL4OPT",
    "nlp4lp_clean": "NLP4LP",
    "complexor_clean": "ComplexOR",
    "resocratic_clean": "ReSocratic",
    "combined": "OPT-Principled",
    "combined_dataset": "OPT-Principled",
    "opt-principled_clean": "OPT-Principled",
}

# Dataset-specific settings (if needed)
DATASET_CONFIG = {
    "ComplexLP": {"tolerance": 0.05},
    "EasyLP": {"tolerance": 0.01},
    "IndustryOR": {"tolerance": 0.05},
    "OPT-Principled": {"tolerance": 0.05},
}

# ============================================
# Utility Functions
# ============================================

def get_prompt_template(template_name="default"):
    """Get prompt template by name"""
    templates = {
        "default": GUROBI_PROMPT,
    }
    return templates.get(template_name, GUROBI_PROMPT)


def get_model_config(model_name):
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, {"temperature": 0.01, "max_tokens": 8192})


def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset"""
    return DATASET_CONFIG.get(normalize_dataset_name(dataset_name), {"tolerance": 0.05})


def normalize_dataset_name(dataset_name: str) -> str:
    """Map historical dataset names to the canonical OPEN benchmark names."""
    if not dataset_name:
        return dataset_name

    name = dataset_name.strip()
    if name.endswith(".jsonl"):
        name = name[:-6]

    alias = DATASET_ALIASES.get(name.casefold())
    if alias:
        return alias

    for canonical_name in DATASETS:
        if canonical_name.casefold() == name.casefold():
            return canonical_name

    if name.endswith("_clean"):
        base_name = name[:-6]
        for canonical_name in DATASETS:
            if canonical_name.casefold() == base_name.casefold():
                return canonical_name

    return name


def get_benchmark_dirs(project_root: Path) -> list[Path]:
    """Return benchmark directories in priority order for the migrated OPEN layout."""
    return [
        project_root.parent.parent / "data" / "benchmarks",
        project_root / "clean_benchmarks",
        project_root.parent / "clean_benchmarks",
    ]


def find_benchmark_path(project_root: Path, dataset_name: str) -> Path:
    """Locate the benchmark file for a dataset, accepting legacy names as input."""
    normalized_name = normalize_dataset_name(dataset_name)
    candidate_names = [normalized_name]
    raw_name = dataset_name[:-6] if dataset_name.endswith(".jsonl") else dataset_name
    if raw_name not in candidate_names:
        candidate_names.append(raw_name)

    for directory in get_benchmark_dirs(project_root):
        for name in candidate_names:
            candidate = directory / f"{name}.jsonl"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Checked directories: "
        f"{[str(path) for path in get_benchmark_dirs(project_root)]}"
    )
