"""Configuration subsystem.

Layered precedence (lowest to highest):
    1. built-in defaults
    2. config.toml
    3. environment variables  (prefix + dotted key, e.g. SYNAPSE_PROVIDERS_OPENAI_API_KEY)
    4. secret files (paths referenced in config/env, content read on demand)

Consumers read settings through ConfigProvider; they never hardcode paths or
know where a value came from.
"""

from synapse.config.provider import ConfigProvider
from synapse.config.settings import Settings

__all__ = ["ConfigProvider", "Settings"]
