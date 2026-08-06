"""Workspace facade — the single component the Master Agent talks to.

Composes the Phase 4 modules (File Manager, Document Parser, Vision Pipeline,
Embedding Engine, Vector Store, Retrieval, Indexer) and exposes:

- ingestion: ``upload`` (dedupe) → ``index`` (background job, progress)
- querying: ``prepare(prompt, file_ids)`` — automatically decides per-file
  pipeline (images → vision, documents → RAG, code → code-aware scan),
  applies the local-only privacy guard, and returns context + outcome.

Phase 5: an optional ``root`` pins every component (files, vector index,
catalog) inside that directory, giving each project an isolated workspace.
"""

from __future__ import annotations

import logging
from pathlib import Path

from synapse.domain.workspace import WorkspaceOutcome
from synapse.events import Events
from synapse.logging import get_logger
from synapse.workspace.config import VectorStoreSettings, WorkspaceSettings
from synapse.workspace.embeddings import EmbeddingEngine
from synapse.workspace.files import WorkspaceFileManager
from synapse.workspace.indexer import WorkspaceIndexer
from synapse.workspace.jobs import JobRegistry
from synapse.workspace.retrieval import Retrieval
from synapse.workspace.vectors import create_vector_store
from synapse.workspace.vision import VisionPipeline

log = get_logger("synapse.workspace")


class Workspace:
    """Owns every workspace component; the public entry is ``prepare``."""

    def __init__(
        self, paths, config, providers, registry, events, memory=None, lifecycle=None, *, root: Path | None = None,
    ) -> None:
        self._settings = WorkspaceSettings.from_config(config.settings.workspace)
        self._vs_settings = VectorStoreSettings.from_config(config.settings.vector_store)
        self._providers = providers
        self._registry = registry
        self._events = events
        self._memory = memory
        self._lifecycle = lifecycle
        self._root = root
        self.enabled = self._settings.enabled

        self.files = WorkspaceFileManager(paths, self._settings, root=self._root)
        self.embeddings = EmbeddingEngine(
            providers, self._settings.embedding_model, self._settings.embed_batch_size
        )
        self.vector_store = create_vector_store(paths, self._vs_settings, root=self._root)
        self.retrieval = Retrieval(self.embeddings, self.vector_store, self.files, self._settings)
        self.vision = VisionPipeline(providers, registry, self._settings, lifecycle=lifecycle)
        self.jobs = JobRegistry()
        self.indexer = WorkspaceIndexer(
            self.files, self.embeddings, self.vector_store, self.vision, memory, events, self._settings
        )
        self._settings = self._settings  # keep for introspection

# -- ingestion ----------------------------------------------------------

    def upload(self, name: str, data: bytes):
        info = self.files.upload(name, data)
        if self._events:
            self._events.publish(Events.FILE_UPLOADED, {"file_id": info.id, "name": info.name})
        return info

    def index(self, file_id: str) -> str:
        return self.indexer.start(file_id, self.jobs)

    def job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def list_files(self):
        return self.files.list()

    def get_file(self, file_id: str):
        return self.files.get(file_id)

    def delete(self, file_id: str) -> bool:
        removed = self.files.delete(file_id)
        if removed:
            self.vector_store.delete(file_id)
        return removed

    # -- querying -----------------------------------------------------------

    def prepare(self, prompt: str, file_ids: list[str] | None = None) -> tuple[str, WorkspaceOutcome]:
        """Resolve attachments → pipeline decision → context for the Master.

        Returns ``(context_text, outcome)``. ``context_text`` is prepended to
        the prompt; the outcome records what was used and whether it was
        forced local-only.
        """
        outcome = WorkspaceOutcome()
        outcome.local_only = not self._settings.allow_cloud_forwarding

        attached = [self.files.get(fid) for fid in (file_ids or [])]
        attached = [f for f in attached if f is not None]
        outcome.files_attached = [f.id for f in attached]

        # 1) Images → Vision Pipeline (answers the question directly).
        for entry in attached:
            if entry.pipeline != "image":
                continue
            data = self.files.read(entry.id)
            if not data:
                continue
            try:
                answer = self.vision.analyze(data, prompt)
                outcome.vision_descriptions.append(answer)
                outcome.pipelines.setdefault("image", []).append(entry.id)
                outcome.files_used.append(entry.id)
                if not entry.vision_description:
                    entry.vision_description = answer[:800]
                    entry.status = "indexed" if entry.indexed_chunks else entry.status
                    self.files.update(entry)
            except Exception as exc:  # noqa: BLE001
                log.warning("vision_failed", file_id=entry.id, error=str(exc)[:120])
        if outcome.vision_descriptions and self._events:
            self._events.publish(Events.VISION_PROCESSED, {"count": len(outcome.vision_descriptions)})

        # 2) Documents/Code → RAG retrieval (+ code scan).
        if self._settings.auto_retrieve:
            text_files = (
                [f for f in attached if f.pipeline in ("document", "code") and f.status == "indexed"]
                if file_ids
                else [f for f in self.files.list() if f.pipeline in ("document", "code") and f.status == "indexed"]
            )
            retrieval_ids = [f.id for f in text_files]
            if retrieval_ids:
                chunks = self.retrieval.retrieve(prompt, self._settings.top_k, retrieval_ids)
                outcome.retrieval = chunks
                context_text = Retrieval.render(chunks)
                outcome.context_chars = len(context_text)
                outcome.used_memory = bool(context_text)
                for entry in text_files:
                    if entry.id in {c.file_id for c in chunks}:
                        outcome.files_used.append(entry.id)
                        outcome.pipelines.setdefault(entry.pipeline, []).append(entry.id)
                if chunks and self._events:
                    self._events.publish(Events.RETRIEVAL_RAN, {"hits": len(chunks)})

            scan_files = (
                [f for f in attached if f.pipeline == "code"]
                if file_ids
                else [f for f in text_files if f.pipeline == "code"]
            )
            triggers = set(self._settings.code_scan_triggers)
            if scan_files and (bool(outcome.files_attached) or any(t in prompt.lower() for t in triggers)):
                outcome.code_matches = self.retrieval.scan_code(
                    self._settings.code_scan_patterns, [f.id for f in scan_files]
                )

        if file_ids:
            outcome.context_chars = max(outcome.context_chars, 0)
        context_text = Retrieval.render(outcome.retrieval)
        if outcome.code_matches:
            lines = [
                f"[{m.file_name}:{m.line}] {m.text}" for m in outcome.code_matches[:50]
            ]
            if lines:
                context_text = (context_text + "\n\n" if context_text else "") + (
                    "Found these markers in your code:\n" + "\n".join(lines)
                )
        return context_text, outcome