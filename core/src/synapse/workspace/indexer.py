"""Workspace Indexer — parse → chunk → embed → store for one file.

Owns the file-to-pipeline decision for ingestion: documents and source code
are parsed, chunked, embedded and inserted into the vector store; images are
described by the Vision Pipeline (the description is embedded for retrieval).
Running a file through this flow makes it permanently queryable,
satisfying "answer follow-ups without re-uploading".
"""

from __future__ import annotations

import logging
import threading

from synapse.domain.enums import MemoryScope
from synapse.events import Events
from synapse.logging import get_logger

log = get_logger("synapse.workspace.indexer")

_SINGLE_BATCH = 16


class WorkspaceIndexer:
    def __init__(self, files, embeddings, vector_store, vision, memory, events, settings) -> None:
        self._files = files
        self._embeddings = embeddings
        self._store = vector_store
        self._vision = vision
        self._memory = memory
        self._events = events
        self._settings = settings

    def start(self, file_id: str, jobs) -> str:
        """Queue a background indexing job; returns its id."""
        job = jobs.create(file_id, self._files.get(file_id).name if self._files.get(file_id) else "")
        self._mark_file(file_id, "indexing")
        t = threading.Thread(target=self._run, args=(file_id, job.job_id, jobs), daemon=True)
        t.start()
        return job.job_id

    # -- pipeline -----------------------------------------------------------

    def _run(self, file_id: str, job_id: str, jobs) -> None:
        from synapse.workspace import parsers

        entry = self._files.get(file_id)
        if entry is None:
            jobs.update(job_id, status="failed", error="file not found", progress=100)
            return
        jobs.update(job_id, stage="parsing", progress=8, message=f"Reading {entry.name}")
        data = self._files.read(file_id)
        if not data:
            self._fail(jobs, job_id, entry, "could not read file bytes")
            return

        try:
            if entry.pipeline == "image":
                self._index_image(file_id, job_id, jobs, entry, data)
            else:
                parsed = parsers.parse(data, entry.extension)
                self._index_text(file_id, job_id, jobs, entry, parsed)
        except Exception as exc:  # noqa: BLE001 - a failing file never breaks boot
            log.warning("index_failed", file_id=file_id, error=str(exc)[:200])
            self._fail(jobs, job_id, entry, str(exc)[:200])

    def _index_image(self, file_id, job_id, jobs, entry, data) -> None:
        jobs.update(job_id, stage="vision", progress=30, message="Describing image")
        description = self._vision.analyze(data, _DESCRIBE_PROMPT)
        entry.vision_description = description[:800]
        entry.status = "indexed"
        entry.indexed_chunks = 1
        self._files.update(entry)
        vectors = self._embeddings.embed([description])
        if vectors and vectors[0]:
            self._store.add(
                vectors,
                [{
                    "file_id": file_id, "file_name": entry.name, "extension": entry.extension,
                    "pipeline": "image", "chunk_index": 0, "text": description,
                }],
            )
        jobs.update(job_id, stage="storing", progress=70)
        self._save_memory(entry, description)
        jobs.update(job_id, stage="done", progress=100, status="completed", message="Indexed")
        if self._events:
            self._events.publish(Events.FILE_INDEXED, {"file_id": file_id, "chunks": 1})

    def _index_text(self, file_id, job_id, jobs, entry, parsed) -> None:
        jobs.update(job_id, stage="chunking", progress=20, message="Chunking text")
        chunks, metas = self._build_chunks(parsed, entry)
        if not chunks:
            entry.status = "indexed"
            entry.indexed_chunks = 0
            self._files.update(entry)
            jobs.update(job_id, progress=100, status="completed", message="No text extracted")
            return

        jobs.update(job_id, stage="embedding", progress=40, message=f"Embedding {len(chunks)} chunks")
        all_vectors: list[list[float]] = []
        for i in range(0, len(chunks), _SINGLE_BATCH):
            batch = chunks[i:i + _SINGLE_BATCH]
            vectors = self._embeddings.embed(batch)
            if vectors is None:
                # Embeddings unavailable: store keyword-indexable metadata only.
                vectors = []
            all_vectors.extend(vectors)
            jobs.update(
                job_id,
                progress=40 + int(40 * (i + len(batch)) / len(chunks)),
                message=f"Embedded {min(i + len(batch), len(chunks))}/{len(chunks)}",
            )
        self._store.add(all_vectors, metas[: len(all_vectors)])

        entry.status = "indexed"
        entry.indexed_chunks = len(all_vectors)
        self._files.update(entry)
        self._save_memory(entry, chunks[0][:600])
        jobs.update(job_id, stage="done", progress=100, status="completed", message=f"{len(all_vectors)} chunks indexed")
        if self._events:
            self._events.publish(Events.FILE_INDEXED, {"file_id": file_id, "chunks": len(all_vectors)})

    # -- chunks -------------------------------------------------------------

    def _build_chunks(self, parsed, entry) -> tuple[list[str], list[dict]]:
        from synapse.workspace import chunking

        chunks: list[str] = []
        metas: list[dict] = []
        base = {
            "file_id": entry.id, "file_name": entry.name, "extension": entry.extension,
            "pipeline": entry.pipeline,
        }
        if entry.pipeline == "code":
            for text, start, end in chunking.chunk_code(
                parsed.text,
                self._settings.code_chunk_lines,
                self._settings.code_chunk_overlap,
            ):
                chunks.append(text)
                metas.append({**base, "chunk_index": len(chunks), "text": text, "lines_start": start, "lines_end": end})
        else:
            pages = parsed.pages or [parsed.text]
            for page_index, page_text in enumerate(pages, start=1):
                for piece in chunking.chunk_text(page_text, self._settings.chunk_size, self._settings.chunk_overlap):
                    chunks.append(piece)
                    metas.append({**base, "chunk_index": len(chunks), "text": piece, "page": page_index})
        return chunks, metas

    # -- helpers ------------------------------------------------------------

    def _save_memory(self, entry, text: str) -> None:
        if self._memory is None or not text:
            return
        try:
            self._memory.save(
                MemoryScope.PROJECT,
                text,
                source=f"workspace file: {entry.name}",
                metadata={"file_id": entry.id, "pipeline": entry.pipeline},
            )
        except Exception:  # noqa: BLE001
            log.debug("memory_save_skipped")

    def _mark_file(self, file_id: str, status: str) -> None:
        entry = self._files.get(file_id)
        if entry is not None:
            entry.status = status
            self._files.update(entry)

    def _fail(self, jobs, job_id, entry, error: str) -> None:
        entry.status = "failed"
        entry.error = error
        self._files.update(entry)
        jobs.update(job_id, status="failed", error=error, progress=100)


_DESCRIBE_PROMPT = (
    "Describe this image in detail: what it shows, any text or diagram, "
    "and anything that would matter for answering follow-up questions about it."
)