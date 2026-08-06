# Synapse One — Python AI Core

The orchestrator "brain" of Synapse One. UI-agnostic, provider-agnostic,
local-first by design. See `../References/Synapse_one.md` for the source
of truth.

## Layout

```
src/synapse/
  config/     # layered configuration (toml + env + secrets)
  contracts/  # abstract interfaces (provider, registry, hardware, config, router, performance)
  domain/     # typed domain entities (incl. weighted ModelCapabilities)
  hardware/   # capability scanner
  providers/  # provider abstraction + vendor implementations
  registry/   # model registry (config-driven)
  analyzers/  # intent / complexity (0-100 calibrated) / privacy
  decision/   # decision engine (capabilities, modality, workspace)
  router/     # Router V2 — intelligent supervisor (Phase 2.5)
  execution/  # planner + executor
  performance/ # performance learning loop (Phase 2.5)
  master/     # Master Agent — the only public entry point
  events/     # pub/sub event bus
  logging/    # structured logging
  di/         # dependency injection container
  bootstrap.py # composition root
```

## Quickstart

```
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"
pytest
```

Phase 2.5: the router is now a capability supervisor (vision exclusion,
specialist weighting, latency prediction, performance learning). See
`../README.md` → "Phase 2.5 — Intelligent Routing Engine".