# -*- coding: utf-8 -*-
"""
Lightweight helpers for categorising optimisation problems and surfacing
category-level memory.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent.parent
DEFAULT_GUIDELINE_PATH = str(_PROJECT_ROOT / "memory_storage" / "category_guidelines.jsonl")


class MemoryIntelligence:
    """
    Heuristic problem classifier + guideline loader.

    The goal is to offer fast, rule-based categorisation that can run
    offline. If the heuristics fail, downstream agents (LLMs) can still
    append tags, but we always return the heuristic view for consistency.
    """

    CATEGORY_KEYWORDS: Dict[str, Set[str]] = {
        "workforce_planning": {
            "worker",
            "workforce",
            "training",
            "trainee",
            "overtime",
            "hire",
            "fire",
        },
        "inventory_planning": {
            "inventory",
            "backlog",
            "stock",
            "warehouse",
            "storage",
            "holding cost",
        },
        "production_planning": {
            "production",
            "factory",
            "capacity",
            "machine",
            "batch",
            "demand",
        },
        "scheduling": {
            "schedule",
            "sequencing",
            "precedence",
            "flow shop",
            "job shop",
            "makespan",
        },
        "transportation": {
            "transport",
            "shipping",
            "vehicle",
            "route",
            "delivery",
            "supply",
            "demand",
            "shipment",
        },
        "network_flow": {
            "flow",
            "arc",
            "network",
            "node",
            "capacity",
            "supply node",
            "demand node",
        },
        "assignment": {
            "assignment",
            "allocate",
            "task",
            "agent",
            "matching",
            "job",
        },
        "facility_location": {
            "facility",
            "location",
            "plant",
            "open",
            "siting",
            "distribution center",
        },
        "traveling_salesman": {
            "tsp",
            "tour",
            "city",
            "travel",
            "route visiting",
            "cyclic",
        },
        "portfolio_optimization": {
            "portfolio",
            "investment",
            "asset",
            "return",
            "risk",
            "variance",
        },
    }

    def __init__(self, guideline_path: str = DEFAULT_GUIDELINE_PATH):
        self.guideline_path = guideline_path
        self.guidelines = self._load_guidelines(guideline_path)

    @staticmethod
    def _load_guidelines(path: str) -> Dict[str, Dict]:
        guidelines: Dict[str, Dict] = {}
        if not path or not os.path.exists(path):
            return guidelines
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                category = payload.get("category")
                if not category:
                    continue
                guidelines[category] = payload
        return guidelines

    def classify(self, description: str, top_k: int = 3, minimum_score: int = 1) -> List[Tuple[str, int]]:
        """
        Return a ranked list of (category, score) using keyword heuristics.
        """
        if not description:
            return []
        text = description.lower()
        scores: Dict[str, int] = defaultdict(int)
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                occurrences = len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text))
                if occurrences:
                    scores[category] += occurrences
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        filtered = [(cat, score) for cat, score in ranked if score >= minimum_score]
        if top_k:
            return filtered[:top_k]
        return filtered

    def categories_only(self, description: str, top_k: int = 3, minimum_score: int = 1) -> List[str]:
        return [cat for cat, _ in self.classify(description, top_k=top_k, minimum_score=minimum_score)]

    def guideline_text(
        self,
        categories: Iterable[str],
        include_header: bool = True,
        max_items_per_category: int = 4,
    ) -> str:
        """
        Render guidelines for the provided categories as a markdown string.
        """
        categories = list(dict.fromkeys(categories))  # deduplicate while preserving order
        if not categories:
            return ""

        lines: List[str] = []
        if include_header:
            lines.append("# Category Playbook")
            lines.append("")

        for category in categories:
            entry = self.guidelines.get(category)
            if not entry:
                continue
            title = entry.get("title") or category.replace("_", " ").title()
            lines.append(f"## {title}")
            guidelines = entry.get("guidelines") or []
            if not guidelines:
                continue
            for bullet in guidelines[:max_items_per_category]:
                lines.append(f"- {bullet}")
            lines.append("")

        return "\n".join(lines).strip()

    def guideline_bullets(self, categories: Iterable[str], max_items_per_category: int = 4) -> List[str]:
        bullets: List[str] = []
        for category in categories:
            entry = self.guidelines.get(category)
            if not entry:
                continue
            title = entry.get("title") or category.replace("_", " ").title()
            guidelines = entry.get("guidelines") or []
            for item in guidelines[:max_items_per_category]:
                bullets.append(f"{title}: {item}")
        return bullets


__all__ = ["MemoryIntelligence", "DEFAULT_GUIDELINE_PATH"]
