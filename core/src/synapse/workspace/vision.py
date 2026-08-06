"""Vision Pipeline — answers questions about images with a vision model.

The image is base64-encoded into a chat message (``images`` field) sent to a
vision-capable model through the standard ModelProvider interface — the same
path any text request uses. The model is resolved from configuration
(``[workspace] vision_model``) with a capability-based fallback; only local
providers are ever consulted for images.
"""

from __future__ import annotations

import base64
import logging

from synapse.domain import ChatMessage, ChatRequest
from synapse.workspace.contracts import VisionPipeline as VisionPipelineContract

log = logging.getLogger("synapse.workspace.vision")


class WorkspaceVisionUnavailable(Exception):
    """Raised when no vision-capable model is installed/configured."""


class VisionPipeline(VisionPipelineContract):
    def __init__(self, providers, registry, settings, lifecycle=None) -> None:
        self._providers = providers
        self._registry = registry
        self._settings = settings
        self._lifecycle = lifecycle

    def analyze(self, data: bytes, prompt: str) -> str:
        model = self._resolve_vision_model()
        provider = self._providers.get(model.provider_id)
        if provider is None:
            raise WorkspaceVisionUnavailable("vision provider unavailable")
        b64 = base64.b64encode(data).decode("ascii")
        request = ChatRequest(
            messages=[
                ChatMessage(role="user", content=prompt, images=[b64]),
            ],
            temperature=0.2,
            model=model.id,
        )
        # Keep the model marked active so the lifecycle manager never unloads
        # it mid-generation (slow CPU inference otherwise gets killed).
        lifecycle = self._lifecycle
        if lifecycle is not None:
            lifecycle.note_request_started(model.provider_id, model.id)
        try:
            response = provider.chat(request)
        finally:
            if lifecycle is not None:
                lifecycle.note_request_completed(model.provider_id, model.id)
        return response.content

    def _resolve_vision_model(self):
        """Pick the configured vision model when installed, else the best
        installed vision-capable model. Configuration wins; never hardcodes."""
        configured = self._settings.vision_model or ""
        installed = self._installed_ids()
        if configured:
            exact = self._registry.get(configured)
            if exact is not None and configured in installed:
                return exact
            for model in self._registry.all():
                if (
                    model.id.split(":")[0] == configured.split(":")[0]
                    and model.capabilities.vision
                    and model.id in installed
                ):
                    return model
        for model in self._registry.all():
            if model.capabilities.vision and model.id in installed:
                return model
        raise WorkspaceVisionUnavailable(
            "no vision-capable model installed (configure vision_model in [workspace])"
        )

    def _installed_ids(self) -> set[str]:
        ids: set[str] = set()
        for provider in self._providers.all():
            try:
                ids.update(d.id for d in provider.list_models())
            except Exception:  # noqa: BLE001
                continue
        return ids