"""Dependency injection container.

Small, typed, composition-root-first container. No global singletons: services
are registered once in ``synapse.bootstrap`` and resolved by interface. Every
subsystem depends on abstractions, never concrete construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from synapse.contracts import ConfigProvider

T = TypeVar("T")


class ServiceNotFound(LookupError):
    """Raised when an interface has no registered implementation."""


class Container:
    """A lazily-constructed service registry.

    Factories are resolved on first access and (optionally) cached. Register
    by interface (e.g. ``ModelRegistry``); resolve by that same interface.
    """

    def __init__(self) -> None:
        self._factories: dict[type, Callable[[], Any]] = {}
        self._singletons: dict[type, Any] = {}
        self._parents: list["Container"] = []

    def register(self, interface: type, factory: Callable[[], Any]) -> None:
        """Register a factory producing an implementation of ``interface``."""
        self._factories[interface] = factory

    def bind_instance(self, interface: type, instance: Any) -> None:
        """Register a pre-built instance (e.g. event bus shared by everyone)."""
        self._singletons[interface] = instance

    def resolve(self, interface: type[T]) -> T:
        if interface in self._singletons:
            return self._singletons[interface]
        factory = self._factories.get(interface)
        if factory is None:
            for parent in self._parents:
                try:
                    return parent.resolve(interface)
                except ServiceNotFound:
                    continue
            raise ServiceNotFound(f"No registered implementation for {interface.__name__}")
        instance = factory()
        self._singletons[interface] = instance
        return instance

    def has(self, interface: type) -> bool:
        return interface in self._factories or interface in self._singletons

    def __contains__(self, interface: type) -> bool:
        return self.has(interface)
