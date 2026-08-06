"""Model lifecycle & resource management (Phase 2.5+).

Keeps loaded models resident only while useful, unloads idle ones, prefers
already-loaded models when the capability gap is small, and reacts to memory
pressure. Provider-agnostic: the actual unload is delegated to the provider when
it supports it (Ollama's ``keep_alive=0``).
"""

from synapse.lifecycle.models import LifecycleSettings, LoadedModel, ModelState

__all__ = [
    "ModelLifecycleManager",
    "ModelLifecycleMetrics",
    "LifecycleSettings",
    "LoadedModel",
    "ModelState",
]

from synapse.lifecycle.manager import ModelLifecycleManager  # noqa: E402
from synapse.lifecycle.metrics import ModelLifecycleMetrics  # noqa: E402
