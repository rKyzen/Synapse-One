"""Application settings — typed, validated view of raw config.

SynapseSettings is the single structured description of what Synapse needs:
paths, logging level, model catalog path, and provider-specific config.
It is produced by the ConfigProvider and consumed by every subsystem through
DI — no module calls `os.getenv` directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr

from synapse.config.paths import SynapsePaths

#: Prefix for environment variable overrides: SYNAPSE_<KEY>.
ENV_PREFIX = "SYNAPSE_"


class LoggingSettings(BaseModel):
    level: str = Field(default="INFO")
    #: true = JSON lines (production), false = human-readable console.
    json_lines: bool = Field(default=False)
    #: Subsystem to silence, e.g. {"httpx": "WARNING"}.
    overrides: dict[str, str] = Field(default_factory=dict)


class ProviderTimeouts(BaseModel):
    """Timeout budget (seconds) for a provider connection.

    ``connect``    — time to establish the TCP connection.
    ``read``       — max idle gap between bytes (i.e. between streamed chunks).
    ``generation`` — total wall-clock budget for a single generation; exceeded
                     mid-stream, the partial output is returned.
    """

    connect: float = Field(default=10.0, gt=0)
    read: float = Field(default=180.0, gt=0)
    generation: float = Field(default=180.0, gt=0)


class ProviderEndpoint(BaseModel):
    """Connection config for a single provider instance."""

    enabled: bool = True
    base_url: str = ""
    #: Environment variable (prefixed automatically) or file path supplying the
    #: secret. Never store the actual secret in config.
    api_key_env: str | None = None
    api_key_file: str | None = None
    #: Differentiated timeout budget. Overridable per-provider in config.toml
    #: under ``[providers.<id>.timeouts]``.
    timeouts: ProviderTimeouts = Field(default_factory=ProviderTimeouts)
    #: Optional extra transport options (proxies, etc.) — free-form.
    options: dict = Field(default_factory=dict)

    def api_key(self, env: dict[str, str] | None = None) -> str | None:
        """Resolve the secret from env var or secret file. Never both."""
        env = env if env is not None else __import__("os").environ
        if self.api_key_env:
            return env.get(self.api_key_env) or env.get(f"{ENV_PREFIX}{self.api_key_env}")
        if self.api_key_file:
            p = __import__("pathlib").Path(self.api_key_file).expanduser()
            return p.read_text().strip() if p.exists() else None
        return None


class PerformanceSettings(BaseModel):
    """Per-model performance history (Phase 2.5 learning loop)."""

    enabled: bool = True
    #: Maximum number of tracked model keys; oldest entries are trimmed.
    max_entries: int = 500


class Settings(BaseModel):
    """Top-level application settings."""

    name: str = "synapse"
    version: str = "0.1.0"
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    #: Map of provider_id -> endpoint config. Providers absent here are skipped.
    providers: dict[str, ProviderEndpoint] = Field(default_factory=dict)
    #: Optional path to an external model catalog TOML. Absent = use inline config.
    model_catalog_file: str | None = None
    #: Inline model catalog entries keyed by model id.
    models: dict[str, dict] = Field(default_factory=dict)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    #: Model lifecycle policy (raw dict; parsed by LifecycleSettings.from_config).
    lifecycle: dict = Field(default_factory=dict)
    #: Task Planner policy (raw dict; parsed by the planner implementation).
    planner: dict = Field(default_factory=dict)
    #: Workspace Memory policy (raw dict; parsed by the memory implementation).
    memory: dict = Field(default_factory=dict)
    #: Phase 4 workspace policy (files, pipelines, chunking, vision, RAG).
    workspace: dict = Field(default_factory=dict)
    #: Phase 4 vector store policy (provider backend).
    vector_store: dict = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict, paths: SynapsePaths | None = None) -> "Settings":
        """Build settings from a raw nested dict (already layered/merged)."""
        return cls.model_validate(raw)
