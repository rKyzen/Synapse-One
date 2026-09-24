"""Synapse One Exact Model Matrix — hard-coded tier & capability mappings.

Non-negotiable architectural rule:
Exactly ONE designated model per capability per hardware tier.
"""

from __future__ import annotations

from synapse.domain.enums import Capability
from synapse.hardware.tier_resolver import HardwareTier


class ModelRole:
    CONVERSATION = "conversation"
    CHAT = "conversation"
    GENERAL = "conversation"
    MATH_REASONING = "math_reasoning"
    MATH = "math_reasoning"
    CODING = "coding"
    PLANNING_AGENT = "planning_agent"
    PLANNING = "planning_agent"
    COMPLEX_TECHNICAL = "complex_technical"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    DEEP_REASONING = "deep_reasoning"


#: Exact, hard-coded model matrix per tier and capability role.
EXACT_MODEL_MATRIX: dict[HardwareTier, dict[str, str]] = {
    HardwareTier.TIER1: {
        ModelRole.CONVERSATION: "gemma3:1b",
        ModelRole.MATH_REASONING: "gemma3:1b",
        ModelRole.CODING: "qwen2.5-coder:1.5b",
        ModelRole.PLANNING_AGENT: "qwen3:1.7b",
        ModelRole.COMPLEX_TECHNICAL: "gemma3:1b",
        ModelRole.VISION: "moondream",
        ModelRole.EMBEDDINGS: "all-minilm",
        ModelRole.DEEP_REASONING: "gemma3:1b",
    },
    HardwareTier.TIER2: {
        ModelRole.CONVERSATION: "gemma3:4b",
        ModelRole.MATH_REASONING: "gemma3:4b",
        ModelRole.CODING: "qwen2.5-coder:7b",
        ModelRole.PLANNING_AGENT: "qwen3:4b",
        ModelRole.COMPLEX_TECHNICAL: "qwen2.5:7b",
        ModelRole.VISION: "qwen2.5vl:7b",
        ModelRole.EMBEDDINGS: "nomic-embed-text",
        ModelRole.DEEP_REASONING: "gemma3:4b",
    },
    HardwareTier.TIER3: {
        ModelRole.CONVERSATION: "gemma3:12b",
        ModelRole.MATH_REASONING: "gemma3:12b",
        ModelRole.CODING: "qwen2.5-coder:14b",
        ModelRole.PLANNING_AGENT: "qwen2.5:14b",
        ModelRole.COMPLEX_TECHNICAL: "qwen2.5:14b",
        ModelRole.VISION: "qwen2.5vl:7b",
        ModelRole.EMBEDDINGS: "nomic-embed-text",
        ModelRole.DEEP_REASONING: "gemma3:12b",
    },
    HardwareTier.TIER3_PLUS: {
        ModelRole.CONVERSATION: "gemma3:12b",
        ModelRole.MATH_REASONING: "gemma3:12b",
        ModelRole.CODING: "qwen2.5-coder:14b",
        ModelRole.PLANNING_AGENT: "qwen2.5:14b",
        ModelRole.COMPLEX_TECHNICAL: "qwen2.5:14b",
        ModelRole.VISION: "qwen2.5vl:7b",
        ModelRole.EMBEDDINGS: "nomic-embed-text",
        ModelRole.DEEP_REASONING: "qwen2.5:32b",
    },
    HardwareTier.CLOUD_FALLBACK: {
        ModelRole.CONVERSATION: "gpt-4o-mini",
        ModelRole.MATH_REASONING: "gpt-4o-mini",
        ModelRole.CODING: "gpt-4o-mini",
        ModelRole.PLANNING_AGENT: "gpt-4o-mini",
        ModelRole.COMPLEX_TECHNICAL: "gpt-4o-mini",
        ModelRole.VISION: "gpt-4o-mini",
        ModelRole.EMBEDDINGS: "all-minilm",
        ModelRole.DEEP_REASONING: "gemini-2.5-flash",
    },
}


def matches_model_id(model_id: str, target_id: str) -> bool:
    """Check if model_id matches target_id allowing for - / _ / :latest variations."""
    m_norm = model_id.lower().replace("-", "").replace("_", "").strip()
    t_norm = target_id.lower().replace("-", "").replace("_", "").strip()

    if m_norm == t_norm:
        return True
    if m_norm == f"{t_norm}:latest" or t_norm == f"{m_norm}:latest":
        return True
    m_base = m_norm[:-7] if m_norm.endswith(":latest") else m_norm
    t_base = t_norm[:-7] if t_norm.endswith(":latest") else t_norm
    if m_base == t_base:
        return True
    if ":" not in t_base and m_base.startswith(f"{t_base}:"):
        return True
    return False


def capability_to_role(cap: Capability | str, *, is_deep_reasoning: bool = False) -> str:
    """Map a Capability or ModelRole string to a ModelRole."""
    cap_val = cap.value if isinstance(cap, Capability) else str(cap).lower()

    if is_deep_reasoning or cap_val in (ModelRole.DEEP_REASONING, "deep_reasoning"):
        return ModelRole.DEEP_REASONING

    if cap_val in (
        ModelRole.CODING, "coding", "code_analysis", "file_creation",
        "file_editing", "testing", "debugging", "terminal", "json", "project_creation"
    ):
        return ModelRole.CODING
    if cap_val in (
        ModelRole.PLANNING_AGENT, "planning", "planning_agent", "architecture",
        "task_management", "automation", "tools"
    ):
        return ModelRole.PLANNING_AGENT
    if cap_val in (
        ModelRole.VISION, "vision", "image_understanding", "ocr", "pdf", "pdf_reading"
    ):
        return ModelRole.VISION
    if cap_val in (
        ModelRole.EMBEDDINGS, "embeddings", "knowledge_retrieval", "memory", "citations"
    ):
        return ModelRole.EMBEDDINGS
    if cap_val in (
        ModelRole.MATH_REASONING, "math", "reasoning", "math_reasoning", "data_analysis", "education"
    ):
        return ModelRole.MATH_REASONING
    if cap_val in (
        ModelRole.COMPLEX_TECHNICAL, "complex_technical", "translation", "languages",
        "document_analysis", "project_analysis"
    ):
        return ModelRole.COMPLEX_TECHNICAL

    return ModelRole.CONVERSATION


def get_exact_model(tier: HardwareTier, cap: Capability | str, *, is_deep_reasoning: bool = False) -> str:
    """Get the exact locked model for a given hardware tier and capability."""
    tier_map = EXACT_MODEL_MATRIX.get(tier, EXACT_MODEL_MATRIX[HardwareTier.TIER1])
    role = capability_to_role(cap, is_deep_reasoning=is_deep_reasoning)
    return tier_map.get(role, "gemma3:1b")

