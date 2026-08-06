"""Provider abstraction — factory and manager.

Only here do vendor classes get instantiated. The Master and the rest of the
system interact with the ProviderManager, which exposes ModelProvider
interfaces only. No vendor name ever appears outside the providers package
(and the config that names them).
"""

from synapse.providers.factory import ProviderFactory
from synapse.providers.manager import ProviderManager

__all__ = ["ProviderFactory", "ProviderManager"]
